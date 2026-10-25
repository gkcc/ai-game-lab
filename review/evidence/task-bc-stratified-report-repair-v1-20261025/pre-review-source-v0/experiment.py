"""One compact-only report repair; the failed source stage remains immutable."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read
from learning.task_bc_stratified_error_sample_v1 import pipeline as p

STAGE = 'task-bc-stratified-report-repair-v1'
SOURCE = ROOT / 'artifacts/task-bc-stratified-error-sample-v1-20261025'
PACKAGE = Path(__file__).resolve().parent
PUBLIC = ROOT.parent / 'publication/ai-game-lab'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PUBLIC)


def stat(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns)


def prepare(out):
    out = Path(out).resolve()
    p.require(not (out / 'registration.json').exists(), 'Registration already exists')
    prior, review = read(SOURCE / 'summary.json'), read(SOURCE / 'evidence-review.json')
    p.require(prior['passed'] is False and review['stage_acceptance'] is False and
              review['evidence_consistency_passed'] is True, 'Failed source stage not verified')
    advice = out / 'independent-next-stage-advice.json'
    p.require(advice.is_file(), 'Independent next-stage advice required')
    protected = read(SOURCE / 'protection-before.json')['stat_before'].copy()
    for path in SOURCE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            protected[path.relative_to(ROOT).as_posix()] = stat(path)
    for path in (ROOT / 'TASK_BC_STRATIFIED_ERROR_SAMPLE_V1.zh-CN.md',):
        protected[path.relative_to(ROOT).as_posix()] = stat(path)
    p.write_new(out / 'protection-before.json', dict(stat_before=protected,
        inherited_protection_sha256=sha(SOURCE / 'protection-before.json'),
        method='Inherited byte hashes remain in the bound predecessor protection. All listed files checked by before/after stat; only exact compact/report/source input allowlists freshly hashed. No selected input arrays or original streams read.'))
    inputs = ['registration.json', 'freeze.json', 'summary.json', 'failure.json',
        'budget-reservations.jsonl', 'evidence-review.json', 'root-evidence-acceptance.json',
        'sample-manifest.json', 'sample-freeze.json', 'report/failure.json',
        'run/job-inputs/report.json', 'run/candidate/aggregate-receipt.json',
        'run/candidate/comparison.json', 'run/candidate/summary.json',
        'run/reference/summary.json', 'run/reference/reference-receipt.json',
        'run/candidate/support-index-receipt.json',
        'independent-review/summary-roundtrip-counterexample.json',
        'independent-review/conditional-recalculation.json']
    for path in sorted((SOURCE / 'run/candidate/shards').iterdir()):
        if path.name.endswith(('.jsonl.gz', '.receipt.json')):
            inputs.append(path.relative_to(SOURCE).as_posix())
    old_cfg = read(SOURCE / 'registration.json')
    from learning.task_entry_repair_v1.experiment import installed_files
    installed = installed_files()
    identities = dict(default_bot=sha(ROOT / 'main.py'), task_model=sha(ROOT / old_cfg['task_model']),
        official_rules=sha(installed['rules']), official_loader=sha(installed['loader']))
    p.require(identities['task_model'] == old_cfg['task_model_sha256'] and
              identities['official_rules'] == old_cfg['engine_sha256'] and
              identities['official_loader'] == old_cfg['official_loader_sha256'], 'Fixed identities changed')
    cfg = dict(schema=STAGE, created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse', 'HEAD').decode().strip(),
        source_stage=SOURCE.relative_to(ROOT).as_posix(), source_stage_acceptance=False,
        source_inputs_sha256={rel:sha(SOURCE / rel) for rel in inputs},
        source_code_sha256=read(SOURCE / 'freeze.json')['runtime_files'],
        identities=identities, model_path=old_cfg['task_model'],
        advice_sha256=sha(advice), protection_sha256=sha(out / 'protection-before.json'),
        prior_budget_ledger_sha256=sha(SOURCE / 'budget-reservations.jsonl'),
        carry_groups=3844, carry_candidates=226982,
        stage_wall_seconds_max=120.0, worker_seconds_max=110.0, cleanup_seconds_max=10.0,
        max_workers=1, blas_threads=1, process_tree_private_commit_bytes=4294967296,
        focused_regression_cases=8, report_entrypoint_calls=1, compact_records_max=514,
        compact_candidates_accounted=32366, new_model_calls=0, evaluated_groups=0,
        evaluated_candidates=0, original_stream_reads=0, numerical_input_decodes=0,
        training_runs=0, policy_calls=0, new_games=0, official_transitions=0,
        independent_review=dict(compact_records_max=514, wall_seconds_max=120,
            model_calls=0, numerical_input_decodes=0, separate_from_execution_clock=True),
        acceptance=['Eight focused comparison regressions pass, including old false rejection and genuine difference rejection.',
            'One report consumes exactly the saved 514 compact records and matches the independently recomputed values.',
            'Six marginals and thirty split/phase/class cells retain counts, identities and declared interpretation limits.',
            'Source stage remains false, source files and budget unchanged, new output only, natural owned tree exit.',
            'Independent review and publication privacy checks pass.'],
        stop='One attempt only. Any failure stops. No source scans, numeric input decode, rescore, replacement sample or training.',
        metadata_hashing_model_bytes_is_not_loading_or_scoring=True,
        clock='Execution clock starts before the sole worker and includes its checks, report and cleanup. Static preparation/review are earlier; independent post-run compact audit has its own 120-second cap.',
        local_registration_not_external_timestamp=True,
        original_diagnostic_stage_acceptance=False, original_policy_admission=False)
    p.write_new(out / 'registration.json', cfg)
    return dict(prepared=True, input_files=len(inputs), protected_stat_files=len(protected))


def freeze(out):
    out = Path(out).resolve(); cfg = read(out / 'registration.json')
    sources = cfg['source_code_sha256'].copy()
    for path in PACKAGE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.md'):
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    published = {rel:sha(PUBLIC / 'kaggriculture' / rel) for rel in sources}
    mappings = read(SOURCE / 'freeze.json')['line_ending_mappings']
    for rel, digest in sources.items():
        p.require(sha(ROOT / rel) == digest, 'Runtime source changed')
        if published[rel] != digest:
            p.require(rel in mappings and
                      (ROOT / rel).read_bytes().replace(b'\r\n', b'\n') ==
                      (PUBLIC / 'kaggriculture' / rel).read_bytes().replace(b'\r\n', b'\n'),
                      'Unregistered source byte difference')
    p.write_new(out / 'freeze.json', dict(registration_sha256=sha(out / 'registration.json'),
        runtime_files=sources, public_source_sha256=published, line_ending_mappings=mappings))
    return dict(frozen=True, sources=len(sources), freeze_sha256=sha(out / 'freeze.json'))


def verify(out):
    cfg = read(out / 'registration.json'); frozen = read(out / 'freeze.json')
    p.require(sha(out / 'registration.json') == frozen['registration_sha256'] and
              sha(out / 'protection-before.json') == cfg['protection_sha256'] and
              sha(out / 'independent-next-stage-advice.json') == cfg['advice_sha256'], 'Repair metadata changed')
    for rel, digest in frozen['runtime_files'].items():
        p.require(sha(ROOT / rel) == digest, 'Frozen source changed: ' + rel)
    for rel, digest in cfg['source_inputs_sha256'].items():
        p.require(sha(SOURCE / rel) == digest, 'Saved source artifact changed: ' + rel)
    from learning.task_entry_repair_v1.experiment import installed_files
    installed = installed_files()
    actual = dict(default_bot=sha(ROOT / 'main.py'), task_model=sha(ROOT / cfg['model_path']),
        official_rules=sha(installed['rules']), official_loader=sha(installed['loader']))
    p.require(actual == cfg['identities'], 'Default/model/official identity changed')
    p.require(read(SOURCE / 'summary.json')['passed'] is False, 'Original failure overwritten')
    return cfg


def worker(out):
    cfg = verify(out); request = read(out / 'request.json')
    p.require(request['freeze_sha256'] == sha(out / 'freeze.json'), 'Worker freeze changed')
    deadline = request['deadline_monotonic']; p.check_deadline(deadline)
    p.write_new(out / 'worker-started.json', dict(request_sha256=sha(out / 'request.json')))
    from .verify_reporting import run_regressions
    tests = run_regressions(deadline=deadline)
    p.require(tests['passed'] and len(tests['cases']) == cfg['focused_regression_cases'], 'Comparison regression failed')
    p.write_new(out / 'focused-regressions.json', tests); p.check_deadline(deadline)
    source_cfg = dict(read(SOURCE / 'registration.json'),
        **read(SOURCE / 'run/job-inputs/report.json')['pipeline_config'])
    source_cfg['repair_freeze_sha256'] = sha(out / 'freeze.json')
    from .reporting import report
    result = report(SOURCE, out, source_cfg, deadline)
    p.require(result['passed'] and result['rows'] == 514 and result['candidates'] == 32366 and
              result['model_calls'] == result['derived_input_decodes'] == result['original_streams_read'] == 0,
              'Report identity or zero-compute constraint failed')
    verify(out); p.check_deadline(deadline)
    p.write_new(out / 'worker-complete.json', dict(passed=True, result=result,
        request_sha256=sha(out / 'request.json'), freeze_sha256=sha(out / 'freeze.json')))


def run(out):
    cfg = verify(out)
    review = read(out / 'pre-review.json'); commit = read(out / 'pre-execution-commit.json')
    p.require(review['passed'] is True and review['freeze_sha256'] == sha(out / 'freeze.json') and
              commit['freeze_sha256'] == sha(out / 'freeze.json') and
              commit['commit'] == git('rev-parse', 'HEAD').decode().strip(), 'Exact source review/commit required')
    for rel, digest in commit['git_files_sha256'].items():
        import hashlib
        p.require(hashlib.sha256(git('show', commit['commit'] + ':' + rel)).hexdigest() == digest,
                  'Committed bytes differ')
    started = time.monotonic(); deadline = started + cfg['stage_wall_seconds_max']
    p.write_new(out / 'started.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=sha(out / 'freeze.json'), deadline_monotonic=deadline, retries_allowed=False))
    work_deadline = min(deadline - cfg['cleanup_seconds_max'], started + cfg['worker_seconds_max'])
    p.write_new(out / 'request.json', dict(freeze_sha256=sha(out / 'freeze.json'), deadline_monotonic=work_deadline))
    from learning.task_bc_diagnostics_capacity_exit_v1.windows_tree import run_owned
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    life, local, failure = {}, {}, None
    try:
        with (out / 'worker.log').open('x', encoding='utf-8') as log:
            code = run_owned([sys.executable, '-B', '-m', 'learning.task_bc_stratified_report_repair_v1.experiment',
                'worker', '--out', str(out)], cwd=ROOT, env=environment, stream=log,
                timeout=work_deadline-time.monotonic(), phase_deadline=deadline,
                lifecycle=life, local_lifecycle=local, memory_limit_bytes=cfg['process_tree_private_commit_bytes'])
        p.require(code == 0 and life['tree_closed'] and life['active_processes_after_close'] == 0 and
                  life['natural_tree_exit'] and not life['termination_requested'] and life['containment_established'] and
                  life['job_handle_closed'] and life['all_process_handles_closed'], 'Worker lifecycle rejected')
        completed = read(out / 'worker-complete.json')
        p.require(completed['passed'] and completed['request_sha256'] == sha(out / 'request.json') and
                  completed['freeze_sha256'] == sha(out / 'freeze.json'), 'Worker completion changed')
        verify(out); p.check_deadline(deadline)
    except BaseException as error:
        failure = dict(kind=type(error).__name__, retry=False)
        raise
    finally:
        protected = read(out / 'protection-before.json')['stat_before']
        changed = [rel for rel, value in protected.items() if not (ROOT / rel).is_file() or stat(ROOT / rel) != value]
        check = dict(passed=not changed, checked_files=len(protected), changed_files=changed,
            method='before/after stat plus fresh exact source and compact allowlist hashes; not full historical byte reread')
        p.write_new(out / 'history-stat-check.json', check)
        p.write_new(out / 'local-lifecycle.json', local); p.write_new(out / 'lifecycle.json', life)
        p.write_new(out / 'summary.json', dict(passed=failure is None and not changed,
            elapsed_seconds=time.monotonic()-started, attempts=1, retries=0, failure=failure,
            freeze_sha256=sha(out / 'freeze.json'), source_stage_acceptance=False,
            original_diagnostic_stage_acceptance=False, original_policy_admission=False,
            source_summary_sha256=sha(SOURCE / 'summary.json'),
            predecessor_budget_sha256=cfg['prior_budget_ledger_sha256'],
            cumulative_reserved_groups=cfg['carry_groups'], cumulative_reserved_candidates=cfg['carry_candidates'],
            new_evaluations=0, new_model_calls=0, input_decodes=0, original_streams_read=0,
            training_runs=0, policy_calls=0, new_games=0, official_transitions=0,
            focused_regressions_sha256=sha(out / 'focused-regressions.json') if (out / 'focused-regressions.json').is_file() else None,
            report_receipt_sha256=sha(out / 'report/receipt.json') if (out / 'report/receipt.json').is_file() else None,
            lifecycle_sha256=sha(out / 'lifecycle.json'), history_stat_check_sha256=sha(out / 'history-stat-check.json')))
        p.require(not changed, 'Protected historical files changed')
    return read(out / 'summary.json')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'freeze', 'run', 'worker'])
    parser.add_argument('--out', required=True); args = parser.parse_args()
    result = globals()[args.command](Path(args.out).resolve())
    if result is not None: print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
