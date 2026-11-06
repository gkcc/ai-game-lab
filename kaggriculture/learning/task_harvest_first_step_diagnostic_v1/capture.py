"""Bounded, whitelisted traceback metadata and one-step private comparisons."""
import ast
from copy import deepcopy
import hashlib
import json
import os

from decision.semantics import canonical
from learning.task_bc_error_diagnostics_v1.shadow import first_difference, ordered_hash


MAX_CONTEXT_BYTES = 384*1024
MAX_LOCATIONS = 16
MAX_TRACE_RECORDS = 128


class FirstStepComplete(RuntimeError):
    """The original first suffix sink returned; never a 48-step success."""


class DiagnosticCapExceeded(RuntimeError):
    """An operation was refused before crossing the single-step boundary."""


class CaptureLimitExceeded(ValueError):
    pass


def bounded_json(value, limit):
    chunks, size = [], 0
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    for text in encoder.iterencode(value):
        raw = text.encode('utf-8')
        size += len(raw)
        if size > limit:
            raise CaptureLimitExceeded()
        chunks.append(raw)
    return b''.join(chunks)


def error_kind(error):
    if type(error) is FirstStepComplete:
        return 'FirstStepComplete'
    if type(error) is DiagnosticCapExceeded:
        return 'DiagnosticCapExceeded'
    if any(cls.__module__ == 'learning.task_anchor_once_v1.storage'
           and cls.__name__ in ('UnsafeWriteError', 'OutputBudgetExceeded') for cls in type(error).__mro__):
        return 'OutputBudgetError'
    for cls in (MemoryError, TimeoutError, OSError, KeyboardInterrupt, SystemExit,
                ValueError, TypeError, AssertionError, KeyError, IndexError, RuntimeError):
        if isinstance(error, cls):
            return cls.__name__
    return 'OtherError'


def _path(value):
    return os.path.normcase(os.path.abspath(os.fspath(value)))


class RegisteredSources:
    """Only authenticated source text creates allowed locations/static sites."""
    def __init__(self, entries):
        self.paths = {}
        self.sites = {}
        for relative, raw, aliases in entries:
            digest = hashlib.sha256(raw).hexdigest()
            tree = ast.parse(raw)
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            code_regions = [(f.name, f.lineno, f.end_lineno) for f in functions]
            comprehensions = {ast.GeneratorExp: '<genexpr>', ast.ListComp: '<listcomp>',
                              ast.DictComp: '<dictcomp>', ast.SetComp: '<setcomp>', ast.Lambda: '<lambda>'}
            code_regions.extend((comprehensions[type(n)], n.lineno, n.end_lineno)
                                for n in ast.walk(tree) if type(n) in comprehensions)
            sites = []
            for node in ast.walk(tree):
                kind, message, exception = None, None, None
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
                    if name == 'require':
                        kind = 'require'
                        if len(node.args) > 1 and isinstance(node.args[-1], ast.Constant) and type(node.args[-1].value) is str:
                            message = node.args[-1].value
                elif isinstance(node, ast.Assert):
                    kind, exception = 'assert', 'AssertionError'
                elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    if node.exc.func.id in ('ValueError', 'TypeError', 'AssertionError', 'RuntimeError'):
                        if node.exc.args and isinstance(node.exc.args[0], ast.Constant) and type(node.exc.args[0].value) is str:
                            kind, exception, message = 'direct_raise', node.exc.func.id, node.exc.args[0].value
                if kind is None:
                    continue
                owners = [f for f in functions if f.lineno <= node.lineno <= f.end_lineno]
                function = min(owners, key=lambda f: f.end_lineno-f.lineno).name if owners else '<module>'
                site_id = 'STATIC_' + hashlib.sha256(
                    f'{relative}:{digest}:{function}:{node.lineno}:{kind}'.encode()).hexdigest()[:20]
                site = dict(id=site_id, kind=kind, function=function, line=node.lineno,
                            end_line=node.end_lineno, message=message, exception=exception)
                sites.append(site)
                self.sites[site_id] = dict(path=relative, source_sha256=digest,
                                          function=function, line=node.lineno,
                                          end_line=node.end_lineno, kind=kind)
            descriptor = dict(path=relative, source_sha256=digest, sites=sites,
                              lines=len(raw.splitlines()), functions=code_regions)
            for alias in aliases:
                self.paths[_path(alias)] = descriptor

    def identify(self, error):
        """Do not stringify error/args; distinguish a raise from a caller only."""
        locations, frames = [], []
        tb = error.__traceback__
        deepest = None
        while tb is not None:
            deepest = tb
            descriptor = self.paths.get(_path(tb.tb_frame.f_code.co_filename))
            if descriptor is not None and 1 <= tb.tb_lineno <= descriptor['lines']:
                name = tb.tb_frame.f_code.co_name
                # Names come from registered code, and no raw path is exported.
                location = dict(path=descriptor['path'], source_sha256=descriptor['source_sha256'],
                                function=name, line=tb.tb_lineno)
                locations.append(location)
                frames.append((tb, descriptor, location))
            tb = tb.tb_next
        locations = locations[-MAX_LOCATIONS:]
        kind = error_kind(error)
        value = dict(kind=kind, id=kind.upper()+'_AT_REGISTERED_CALLSITE',
                     location_kind='registered_callsite_only' if locations else 'unlocated',
                     static_site_id=None, location=locations[-1] if locations else None,
                     locations=locations, rejected_value_kind=None)
        if (frames and frames[-1][0] is deepest
                and frames[-1][1]['path'] == 'kaggriculture/learning/task_request_gate_v1/policy.py'
                and deepest.tb_frame.f_code.co_name == '_typed'):
            value['rejected_value_kind'] = container_kind(deepest.tb_frame.f_locals.get('value'))
        if kind == 'FirstStepComplete':
            value.update(id='FIRST_SUFFIX_SINK_COMPLETED', location_kind='controlled_stop')
            return value
        if kind == 'DiagnosticCapExceeded':
            value.update(id='SINGLE_STEP_CAP_REFUSED', location_kind='diagnostic_cap')
            return value
        last_registered_is_raiser = bool(frames) and frames[-1][0] is deepest
        if last_registered_is_raiser:
            tb, descriptor, location = frames[-1]
            candidates = [s for s in descriptor['sites'] if s['line'] <= tb.tb_lineno <= s['end_line']
                          and s['kind'] in ('assert', 'direct_raise') and s['exception'] == kind]
            if candidates:
                site = min(candidates, key=lambda s: s['end_line']-s['line'])
                if site['message'] is None or error.args == (site['message'],):
                    value.update(id=site['id'], static_site_id=site['id'], location_kind=site['kind'])
                    return value
            # A known require helper raises a ValueError at a dynamic message;
            # only its actual registered caller's require site is attributable.
            if kind == 'ValueError' and location['function'] == 'require' and len(frames) >= 2:
                caller, desc, at = frames[-2]
                candidates = [s for s in desc['sites'] if s['kind'] == 'require'
                              and s['line'] <= caller.tb_lineno <= s['end_line']]
                if candidates:
                    site = min(candidates, key=lambda s: s['end_line']-s['line'])
                    if site['message'] is None or error.args == (site['message'],):
                        value.update(id=site['id'], static_site_id=site['id'], location_kind='require', location=at)
        return value

    def validate_location(self, error):
        if (type(error) is not dict or set(error) != {'kind', 'id', 'location_kind',
                'static_site_id', 'location', 'locations', 'rejected_value_kind'}
                or type(error['locations']) is not list or len(error['locations']) > MAX_LOCATIONS
                or error['rejected_value_kind'] not in (None, *CONTAINER_KINDS)):
            raise ValueError('Diagnostic location schema differs')
        if error['static_site_id'] is not None:
            site = self.sites.get(error['static_site_id'])
            if site is None or error['id'] != error['static_site_id'] or error['location_kind'] != site['kind']:
                raise ValueError('Diagnostic static site differs')
            at = error['location']
            if any(at[key] != site[key] for key in ('path', 'source_sha256', 'function')):
                raise ValueError('Diagnostic static location differs')
            if type(at['line']) is not int or not site['line'] <= at['line'] <= site['end_line']:
                raise ValueError('Diagnostic static line differs')
            if at not in error['locations']:
                raise ValueError('Diagnostic static frame is absent')
        allowed = {(v['path'], v['source_sha256']): v for v in self.paths.values()}
        for location in error['locations']:
            if type(location) is not dict or set(location) != {'path', 'source_sha256', 'function', 'line'}:
                raise ValueError('Diagnostic frame schema differs')
            source = allowed.get((location['path'], location['source_sha256']))
            if source is None or type(location['line']) is not int or not 1 <= location['line'] <= source['lines']:
                raise ValueError('Diagnostic location is outside registered sources')
            if (location['function'] != '<module>' and not any(
                    name == location['function'] and start <= location['line'] <= end
                    for name, start, end in source['functions'])):
                raise ValueError('Diagnostic function is outside registered source')
        if error['location'] is not None and error['location'] not in error['locations']:
            raise ValueError('Diagnostic selected frame is absent')
        if error['static_site_id'] is None:
            expected = ('FIRST_SUFFIX_SINK_COMPLETED' if error['location_kind'] == 'controlled_stop'
                        else 'SINGLE_STEP_CAP_REFUSED' if error['location_kind'] == 'diagnostic_cap'
                        else error['kind'].upper()+'_AT_REGISTERED_CALLSITE')
            if error['id'] != expected:
                raise ValueError('Diagnostic nonstatic identifier differs')


def frame_comparison(actual, expected, digest):
    def ordered_view(frame):
        return dict(frame=canonical(frame), shed_order=[list(player['observation']['private']['shed'].items())
                                                        for player in frame])
    return dict(actual_semantic_sha256=digest(actual), expected_semantic_sha256=digest(expected),
        actual_ordered_sha256=ordered_hash(actual), expected_ordered_sha256=ordered_hash(expected),
        first_semantic_difference=first_difference(ordered_view(actual), ordered_view(expected)))


CONTAINER_KINDS = ('dict_exact', 'dict_subclass', 'official_Struct', 'list_exact',
                   'list_subclass', 'tuple_exact', 'scalar', 'other')


def container_kind(value):
    if type(value) is dict:
        return 'dict_exact'
    if isinstance(value, dict):
        if type(value).__name__ == 'Struct' and type(value).__module__ == 'kaggle_environments.utils':
            return 'official_Struct'
        return 'dict_subclass'
    if type(value) is list:
        return 'list_exact'
    if isinstance(value, list):
        return 'list_subclass'
    if type(value) is tuple:
        return 'tuple_exact'
    if type(value) in (str, int, float, bool, type(None)):
        return 'scalar'
    return 'other'


class Collector:
    def __init__(self, registry):
        self.registry = registry
        self.error = None
        self.context = None
        self.capture = dict(status='not_entered', bytes=0, sha256=None)
        self.suffix_sink_completed = False

    def record(self, error, *, core, target, replay, runtime):
        """Capture failure is evidence; it never replaces the original exception."""
        try:
            self.error = self.registry.identify(error)
            current, tb = None, error.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code is core._run_arm.__code__:
                    current = tb.tb_frame
                tb = tb.tb_next
            if current is None:
                raise ValueError('No registered arm frame')
            # Whitelist access only; never copy/serialize f_locals itself.
            values = current.f_locals
            if values.get('mode') != 'control' or values.get('step') != 240 or values.get('offset') != 0:
                raise ValueError('Outside first factual step')
            original, submitted = values['original'], values['request']
            tracer, environment = values['tracer'], values['environment']
            records, actual, expected = tracer.records, environment.state, replay['steps'][241]
            if type(records) is not list or len(records) > MAX_TRACE_RECORDS:
                raise CaptureLimitExceeded()
            comparison = frame_comparison(actual, expected, core.frame_digest)
            accepted_action = actual[target['side']]['action']
            try:
                equal = core.same_request(accepted_action, submitted)
                request_check = dict(outcome='equal' if equal else 'different', error_kind=None)
            except (MemoryError, TimeoutError, OSError, KeyboardInterrupt, SystemExit):
                raise
            except BaseException as comparison_error:
                request_check = dict(outcome='raises', error_kind=error_kind(comparison_error))
            context = dict(phase='first_factual_step', step=240, side=target['side'],
                original_request=original, submitted_request=submitted, tracer_records=records,
                actual_after_frame=actual, expected_after_frame=expected, comparison=comparison,
                target_stable_trace_sha256=ordered_hash(runtime.stable(values['trace'])),
                target_model_counts=values['trace']['model_counts'],
                target_turn_model_counts=values['trace']['turn_model_counts'],
                interpreter_restored=environment.interpreter is runtime.interpreter,
                target_request_container_kinds=dict(original=container_kind(original),
                    submitted=container_kind(submitted), actual_accepted=container_kind(accepted_action)),
                original_type_request_comparison=request_check,
                serialized_types_note='JSON retains content/order, not Python dict subclasses; not proof same_request passed')
            raw = bounded_json(context, MAX_CONTEXT_BYTES)
            self.context = json.loads(raw)
            self.capture = dict(status='complete', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        except BaseException as capture_error:
            self.context = None
            self.capture = dict(status='limit_exceeded' if isinstance(capture_error, CaptureLimitExceeded)
                                else 'capture_failed', bytes=0, sha256=None)

    def snapshot(self):
        return dict(error=deepcopy(self.error), context=deepcopy(self.context),
                    capture=deepcopy(self.capture), suffix_sink_completed=self.suffix_sink_completed)
