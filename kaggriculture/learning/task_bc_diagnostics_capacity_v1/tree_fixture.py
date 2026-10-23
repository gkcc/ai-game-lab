"""Real bounded process fixtures; deliberately imports no data/model/game code.

Only the registered verify_tree run should invoke this script. Each worker tree
has a shared five-second safety deadline. Ignoring the cooperative deadline does
not mean ignoring that final fixture safety bound.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learning.task_bc_diagnostics_capacity_v1.windows_tree import OwnedTree, current_identity


SAFETY_EXIT = 99
COOPERATIVE_EXIT = 23


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()


def identity_receipt(role):
    return dict(current_identity(), role=role, argv=list(sys.argv), executable=sys.executable,
                parent_pid=os.getppid(), observed_monotonic=time.monotonic())


def child_command(args, depth):
    return [sys.executable, "-B", str(Path(__file__).resolve()), "tree", "--out", str(args.out),
            "--mode", args.mode, "--depth", str(depth), "--deadline", repr(args.deadline),
            "--safety-until", repr(args.safety_until)]


def tree_worker(args):
    write_new(args.out / f"node-{args.depth}.json", identity_receipt(f"node-{args.depth}"))
    child = None
    if args.depth < 2:
        child = subprocess.Popen(child_command(args, args.depth + 1),
            stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8"))
    if args.mode == "normal":
        if child is not None:
            code = child.wait(timeout=max(0.01, args.safety_until - time.monotonic()))
            if code:
                return code
        else:
            time.sleep(0.20)
        code = 0
    elif args.mode == "cooperative":
        while time.monotonic() < args.deadline:
            time.sleep(0.02)
        if child is not None:
            code = child.wait(timeout=max(0.01, args.safety_until - time.monotonic()))
            if code != COOPERATIVE_EXIT:
                return code or 98
        code = COOPERATIVE_EXIT
    else:
        # This intentionally does not honor args.deadline. The tested job owner
        # must terminate it and its descendants before the safety bound fires.
        while time.monotonic() < args.safety_until:
            time.sleep(0.02)
        write_new(args.out / f"node-{args.depth}-safety-fired.json", {"safety_fired": True})
        return SAFETY_EXIT
    write_new(args.out / f"node-{args.depth}-done.json", {"exit_code": code})
    return code


def controller(args):
    """Own a nested worker job; the outer regression kills this exact interpreter.

    No outside handle to the inner job is ever opened. Otherwise an observer
    could inadvertently keep the last job handle alive and defeat this test.
    """
    write_new(args.out / "controller.json", identity_receipt("controller"))
    lifecycle = {}
    tree = OwnedTree(lifecycle)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "tree", "--out",
        str(args.out / "workers"), "--mode", "ignore", "--deadline", repr(args.deadline),
        "--safety-until", repr(args.safety_until)]
    try:
        with (args.out / "inner-worker.log").open("x", encoding="utf-8") as stream:
            tree.start(command, cwd=Path(__file__).resolve().parents[2], env=os.environ, stream=stream)
            write_new(args.out / "inner-established.json", dict(lifecycle))
            while time.monotonic() < args.safety_until:
                tree.sample()
                code = tree.poll()
                if code is not None:
                    return code
                time.sleep(0.02)
            return SAFETY_EXIT
    finally:
        try:
            tree.close()
        finally:
            write_new(args.out / "controller-finally.json", lifecycle)


def sentinel(args):
    write_new(args.out / "sentinel.json", identity_receipt("sentinel"))
    while time.monotonic() < args.safety_until:
        if (args.out / "stop-sentinel").exists():
            return 0
        time.sleep(0.05)
    return SAFETY_EXIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("tree", "controller", "sentinel"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "cooperative", "ignore"), default="normal")
    parser.add_argument("--depth", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--deadline", type=float, default=0.0)
    parser.add_argument("--safety-until", type=float, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.safety_until <= time.monotonic():
        return SAFETY_EXIT
    if args.kind == "tree":
        return tree_worker(args)
    if args.kind == "controller":
        return controller(args)
    return sentinel(args)


if __name__ == "__main__":
    raise SystemExit(main())
