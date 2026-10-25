"""Meter the two original model constructors, including SHA and lazy NPZ reads.

Only registered model files are wrapped.  Their reads are unbuffered, bounded,
and charged on return, including reads after seek.  No replacement loader or
numerical path is introduced.  All temporary patches are restored in finally.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import io
import math
import os
from pathlib import Path
import re
import time


class ModelIOStopped(ValueError):
    """A model-read admission or accounting check failed; no retry is allowed."""


_COUNTERS = (
    "checkpoint_hash_passes_started", "checkpoint_hash_passes_completed",
    "model_object_loads_reserved", "model_object_loads_started", "model_object_loads_completed",
    "model_object_loads_failed", "np_load_calls_started", "np_load_calls_completed",
    "model_opens_started", "model_opens_completed", "model_read_calls_started",
    "model_read_calls_completed", "total_model_bytes_returned_by_reads",
    "hash_bytes_returned_by_reads", "np_load_bytes_returned_by_reads",
    "model_seeks_started", "model_seeks_completed", "model_closes_started", "model_closes_completed",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ModelIOStopped(message)


def _identity(stat):
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev, inode=stat.st_ino)


class _MeteredFile:
    """Minimal seekable binary stream; no fileno/raw/read bypass is exposed."""
    def __init__(self, raw, meter, name, phase, path):
        self._raw, self._meter, self._name, self._phase, self._path = raw, meter, name, phase, path

    @property
    def closed(self):
        return self._raw.closed

    @property
    def name(self):
        return str(self._path)

    @property
    def mode(self):
        return "rb"

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def _read(self, method, size=-1, target=None):
        meter = self._meter
        meter.check()
        _require(not self.closed and meter.active == self._name and meter.phase == self._phase,
                 "model read outside its registered constructor phase")
        _require(type(size) is int, "invalid model read size")
        room = meter.limit["total_model_bytes_returned_by_reads_max"] - meter.counts["total_model_bytes_returned_by_reads"]
        _require(size == 0 or room > 0, "model byte budget exhausted before read")
        amount = min(room, size if size >= 0 else room)
        meter.charge(self._name, model_read_calls_started=1)
        try:
            if target is None:
                result = getattr(self._raw, method)(amount)
                returned = len(result)
            else:
                result = self._raw.readinto(memoryview(target)[:amount])
                returned = result
        except BaseException as exc:
            meter.unknown["failures"].append(dict(component="model_io", model=self._name,
                quantity="model_bytes_in_failed_read", value=None, reason=type(exc).__name__,
                requested_bytes=amount, phase=self._phase))
            raise
        meter.charge(self._name, model_read_calls_completed=1,
                     total_model_bytes_returned_by_reads=returned,
                     **{self._phase + "_bytes_returned_by_reads": returned})
        meter.check()
        return result

    def read(self, size=-1):
        return self._read("read", size)

    def read1(self, size=-1):
        return self.read(size)

    def readall(self):
        return self.read()

    def readline(self, size=-1):
        return self._read("readline", size)

    def readinto(self, target):
        view = memoryview(target).cast("B")
        return self._read("readinto", len(view), view)

    def readinto1(self, target):
        return self.readinto(target)

    def seek(self, offset, whence=0):
        self._meter.check()
        _require(not self.closed and self._meter.active == self._name,
                 "model seek outside its registered constructor")
        self._meter.charge(self._name, model_seeks_started=1)
        result = self._raw.seek(offset, whence)
        self._meter.charge(self._name, model_seeks_completed=1)
        return result

    def tell(self):
        return self._raw.tell()

    def close(self):
        if self.closed:
            return
        tick = time.monotonic()
        meter = self._meter
        meter.charge(self._name, model_closes_started=1)
        error = None
        try:
            identity = _identity(os.fstat(self._raw.fileno()))
            _require(identity == meter._identities[self._name] == _identity(self._path.stat()),
                     "model file changed during read")
            meter.per[self._name]["file_identity_checked"] = True
        except BaseException as exc:
            meter.per[self._name]["file_identity_checked"] = False
            error = exc
        finally:
            try:
                self._raw.close()
                meter.charge(self._name, model_closes_completed=1)
            except BaseException as exc:
                error = error or exc
        if time.monotonic() - tick > meter.close_seconds:
            error = error or ModelIOStopped("model close time budget exhausted")
        if error is not None:
            meter.per[self._name]["close_failure"] = type(error).__name__
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as close_error:
            if exc is None:
                raise
            exc.add_note("Model close/stat verification also failed: " + str(close_error))
        return False


class _ModelMeter:
    def __init__(self, root, cfg, ledger, deadline):
        _require("model_io" not in ledger, "model construction already registered; no retry")
        self.limit = cfg["budget"]["models"]
        for key in ("distinct_npz_files_max", "checkpoint_hash_passes_max", "model_object_loads_max", "np_load_calls_max"):
            _require(type(self.limit.get(key)) is int and self.limit[key] == 2, "two model admissions required: " + key)
        _require(type(self.limit.get("total_model_bytes_returned_by_reads_max")) is int
                 and 0 < self.limit["total_model_bytes_returned_by_reads_max"] <= 4 * 1024 * 1024,
                 "invalid registered model read budget")
        _require(type(deadline) in (int, float) and math.isfinite(deadline), "invalid model deadline")
        self.deadline = min(deadline, time.monotonic() + cfg["budget"]["model_load_phase_seconds_max"])
        self.close_seconds = cfg["budget"]["per_close_seconds_max"]
        self.models = cfg["models"]
        _require(set(self.models) == {"task", "crop"}, "only the two registered models are allowed")
        self.paths = {}
        for name, spec in self.models.items():
            _require(isinstance(spec.get("path"), str) and isinstance(spec.get("sha256"), str)
                     and _SHA.fullmatch(spec["sha256"]), "invalid registered model identity")
            path = (root / spec["path"]).resolve()
            _require(path.is_relative_to(root) and path.suffix.lower() == ".npz", "invalid registered model path")
            self.paths[name] = path
        _require(len(set(self.paths.values())) == 2, "registered model paths must differ")
        self.counts = {key: 0 for key in _COUNTERS}
        self.counts.update(status="started", distinct_npz_files=0, blocked_operations=0, per_model={})
        self.per = self.counts["per_model"]
        for name, spec in self.models.items():
            self.per[name] = {key: 0 for key in _COUNTERS}
            self.per[name].update(path=spec["path"], expected_sha256=spec["sha256"],
                                  sha_verified=False, file_identity_checked=False, status="reserved")
        self._identities = {}  # File-system identities never enter the receipt.
        ledger["model_io"] = self.counts
        self.unknown = ledger.setdefault("unknown_accounting", {"planned_prefix": [], "failures": []})
        _require(isinstance(self.unknown, dict) and all(
            isinstance(self.unknown.get(key), list) for key in ("planned_prefix", "failures")),
            "invalid unknown-accounting ledger")
        self.handles = []
        self.active = self.phase = None
        self.opened = set()
        for name in self.models:
            self.charge(name, model_object_loads_reserved=1)

    def charge(self, name, **amounts):
        for key, value in amounts.items():
            self.counts[key] += value
            self.per[name][key] += value

    def check(self):
        _require(time.monotonic() < self.deadline, "model load deadline exhausted")
        _require(self.counts["total_model_bytes_returned_by_reads"] <= self.limit["total_model_bytes_returned_by_reads_max"],
                 "model byte budget exceeded")

    def identify(self, value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        try:
            candidate = Path(os.fsdecode(value)).resolve()
        except (TypeError, ValueError, OSError):
            return None
        return next((name for name, path in self.paths.items() if path == candidate), None)

    def require_name(self, value, expected=None):
        name = self.identify(value)
        if name is None or (expected is not None and name != expected):
            self.counts["blocked_operations"] += 1
            raise ModelIOStopped("unregistered model path or wrong constructor model")
        return name

    def wrap_open(self, original):
        def opened(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
            name = self.identify(file)
            if name is None:
                if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).suffix.lower() == ".npz":
                    self.counts["blocked_operations"] += 1
                    raise ModelIOStopped("unregistered NPZ open")
                return original(file, mode, buffering, encoding, errors, newline, closefd, opener)
            self.check()
            _require(mode == "rb" and encoding is None and errors is None and newline is None
                     and closefd is True and opener is None, "only ordinary read-only model opens are allowed")
            _require(self.active == name and self.phase in ("hash", "np_load"), "model open outside registered load phase")
            phase_key = self.phase + "_opens_started"
            _require(self.per[name].get(phase_key, 0) == 0, "extra model open is forbidden")
            self.per[name][phase_key] = 1
            self.per[name]["file_identity_checked"] = False
            self.charge(name, model_opens_started=1)
            raw = original(file, "rb", buffering=0)
            self.charge(name, model_opens_completed=1)
            handle = _MeteredFile(raw, self, name, self.phase, self.paths[name])
            self.handles.append(handle)  # Own the handle before any later validation can fail.
            if name not in self.opened:
                self.opened.add(name)
                self.counts["distinct_npz_files"] += 1
            identity = _identity(os.fstat(raw.fileno()))
            if name not in self._identities:
                self._identities[name] = identity
            _require(identity == self._identities[name] == _identity(self.paths[name].stat()),
                     "model changed between registered reads")
            self.check()
            return handle
        return opened

    def wrap_sha(self, name, original):
        def sha(path):
            self.check()
            self.require_name(path, name)
            _require(self.active == name and self.phase is None
                     and self.per[name]["checkpoint_hash_passes_started"] == 0, "extra or out-of-order model hash")
            self.charge(name, checkpoint_hash_passes_started=1)
            self.phase = "hash"
            try:
                digest = original(path)
                self.charge(name, checkpoint_hash_passes_completed=1)
                self.per[name]["actual_sha256"] = digest
                _require(digest == self.models[name]["sha256"], "registered model SHA differs")
                self.per[name]["sha_verified"] = True
                self.check()
                return digest
            finally:
                self.phase = None
        return sha

    def wrap_np_load(self, original):
        def load(file, *args, **kwargs):
            self.check()
            name = self.require_name(file, self.active)
            _require(self.active == name and self.phase is None and not args
                     and kwargs == {"allow_pickle": False}, "unexpected np.load arguments or phase")
            _require(self.per[name]["sha_verified"] and self.per[name]["np_load_calls_started"] == 0,
                     "extra or out-of-order np.load")
            self.charge(name, np_load_calls_started=1)
            self.phase = "np_load"  # Remains active for the returned NpzFile's lazy reads.
            result = original(file, **kwargs)
            self.charge(name, np_load_calls_completed=1)
            self.check()
            return result
        return load

    def wrap_init(self, name, original):
        def initialize(instance, path, expected_sha):
            self.check()
            self.require_name(path, name)
            _require(expected_sha == self.models[name]["sha256"], "constructor expected SHA differs")
            _require(self.active is None and self.per[name]["model_object_loads_started"] == 0,
                     "extra or nested model object load")
            self.charge(name, model_object_loads_started=1)
            self.per[name]["status"] = "started"
            self.active, self.phase = name, None
            try:
                original(instance, path, expected_sha)
                self.charge(name, model_object_loads_completed=1)
                self.per[name]["status"] = "completed"
                self.check()
            except BaseException as exc:
                self.charge(name, model_object_loads_failed=1)
                self.per[name].update(status="failed", failure_type=type(exc).__name__, failure=type(exc).__name__)
                raise
            finally:
                self.active = self.phase = None
        return initialize

    def validate_completed(self):
        self.check()
        _require(self.counts["distinct_npz_files"] == 2, "both registered model files were not opened")
        for name, per in self.per.items():
            _require(all(per[key] == 1 for key in (
                "model_object_loads_reserved", "model_object_loads_started", "model_object_loads_completed",
                "checkpoint_hash_passes_started", "checkpoint_hash_passes_completed",
                "np_load_calls_started", "np_load_calls_completed", "hash_opens_started", "np_load_opens_started")),
                "incomplete or extra model constructor consumption: " + name)
            _require(per["sha_verified"] and per["file_identity_checked"] and per["model_object_loads_failed"] == 0
                     and per["model_opens_started"] == per["model_opens_completed"] == 2
                     and per["model_closes_started"] == per["model_closes_completed"] == 2
                     and per["model_read_calls_started"] == per["model_read_calls_completed"]
                     and per["model_seeks_started"] == per["model_seeks_completed"], "incomplete model I/O: " + name)
        _require(all(handle.closed for handle in self.handles), "model constructor retained an open file")


@contextmanager
def model_reads(root: Path, cfg: dict, ledger: dict, deadline: float):
    """Yield None around exactly one original JointAgent construction.

    Object reservations, starts and completions are separate.  NPZ completion
    counts numpy.load returning; object completion also requires all lazy array
    reads and the original constructor's validations to finish successfully.
    """
    meter = _ModelMeter(Path(root).resolve(), cfg, ledger, deadline)
    tick = time.monotonic()
    originals = None
    error = None
    try:
        meter.check()
        # Imports do not construct models. Keep the patches scoped to this context.
        from learning.task_bc_v1 import model as task_model
        from learning.bc_crop import agent as crop_agent
        import numpy as np
        originals = (builtins.open, io.open, task_model.sha, crop_agent.sha, np.load,
                     task_model.TaskNetwork.__init__, crop_agent.Network.__init__)
        meter.check()
        builtins.open = meter.wrap_open(originals[0])
        io.open = meter.wrap_open(originals[1])
        task_model.sha = meter.wrap_sha("task", originals[2])
        crop_agent.sha = meter.wrap_sha("crop", originals[3])
        np.load = meter.wrap_np_load(originals[4])
        task_model.TaskNetwork.__init__ = meter.wrap_init("task", originals[5])
        crop_agent.Network.__init__ = meter.wrap_init("crop", originals[6])
        yield None
        meter.validate_completed()
        meter.counts["status"] = "completed"
    except BaseException as exc:
        error = exc
        meter.counts.update(status="failed", failure_type=type(exc).__name__, failure=type(exc).__name__)
        raise
    finally:
        cleanup_error = None
        try:
            for handle in meter.handles:
                if not handle.closed:
                    try:
                        handle.close()
                    except BaseException as exc:
                        cleanup_error = cleanup_error or exc
        finally:
            if originals is not None:
                (builtins.open, io.open, task_model.sha, crop_agent.sha, np.load,
                 task_model.TaskNetwork.__init__, crop_agent.Network.__init__) = originals
            meter.counts["patches_restored"] = True
            meter.counts["open_handles_remaining"] = sum(not handle.closed for handle in meter.handles)
            meter.counts["elapsed_seconds"] = time.monotonic() - tick
        if cleanup_error is not None:
            meter.counts.update(status="failed", cleanup_failure_type=type(cleanup_error).__name__,
                                cleanup_failure=type(cleanup_error).__name__)
            if error is None:
                raise cleanup_error
            error.add_note("Model context cleanup also failed: " + str(cleanup_error))
