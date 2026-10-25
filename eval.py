import json
import logging
import os
from functools import partial
from pathlib import Path


def benchmark_failures(run, run_dir, model_name, datasets, expected_indices, mode):
    """Validate this invocation's status and outputs, including swallowed errors."""
    try:
        status = json.loads((run_dir / 'status.json').read_text(encoding = 'utf-8'))
        if status.get('model_name') != model_name or status.get('eval_id') != run_dir.name:
            raise ValueError('status does not identify the current model and run')
        records = status['datasets']
        if not isinstance(records, dict):
            raise ValueError('dataset statuses must be a mapping')
        reports = {row['benchmark']: row for row in run.collect_run_benchmark_report(run_dir)}
    except Exception as exc:
        return [f'{model_name}: cannot validate {run_dir}: {exc}']

    failures = []
    for dataset_name in datasets:
        label = f'{model_name} / {dataset_name}'
        record = records.get(dataset_name)
        if not isinstance(record, dict):
            failures.append(f'{label}: missing dataset status')
            continue
        if 'error_message' in record:
            failures.append(f'{label}: {record["error_message"] or "unspecified evaluation error"}')
            continue
        if record.get('status') != 'done':
            failures.append(f'{label}: incomplete status {record.get("status")!r}')
            continue

        skip_reason = record.get('skip_reason')
        inference_only_reasons = {
            'mode_infer',
            'official_submission_only_mmmu_test',
            'official_submission_only_mmt_bench',
        }
        if skip_reason and not (mode == 'infer' and skip_reason in inference_only_reasons):
            failures.append(f'{label}: skipped ({skip_reason})')
            continue
        if mode != 'infer' and (not isinstance(record.get('metrics'), dict) or not record['metrics']):
            failures.append(f'{label}: missing scoring results')

        indices = expected_indices.get((model_name, dataset_name))
        if not indices:
            failures.append(f'{label}: no prepared dataset indices')
            continue
        try:
            # The status field may contain a cwd-relative path. Reconstruct the
            # prediction path from the exact run directory instead.
            prediction_file = run.get_pred_file_path(str(run_dir), model_name, dataset_name, use_env_format = True)
            if not run.is_prediction_complete(prediction_file, dataset_indices = indices, retry_failed = True):
                failures.append(f'{label}: missing, incomplete, or failed predictions')

            report = reports.get(dataset_name)
            if report is None:
                failures.append(f'{label}: missing benchmark report')
                continue
            fields = ('infer_failed',) if mode == 'infer' else ('infer_failed', 'judge_failed')
            for field in fields:
                count = report.get(field)
                if count is not None and count > 0:
                    failures.append(f'{label}: {field}={count}')
        except Exception as exc:
            failures.append(f'{label}: cannot validate predictions or scoring: {exc}')

    return failures


def use_completion_tokens(wrapper):
    """Give GPT-6 judges the request format VLMEvalKit reserves for o-series and GPT-5 models."""
    original_init = wrapper.__init__

    def init(self, *positional, **kwargs):
        original_init(self, *positional, **kwargs)
        # These models reject max_tokens and any temperature other than the default.
        if 'gpt-6' in self.model:
            self.is_max_completion_tokens = True

    wrapper.__init__ = init


def run_checked(run, args, distributed):
    """Finish toolkit cleanup, then make recorded benchmark failures fatal."""
    from vlmeval.api.base import BaseAPI

    run_dirs = {}
    expected_indices = {}
    judge_fallbacks = []
    current_model = None
    current_dataset = None
    original_status = run.upsert_run_status
    original_build = run.build_dataset_from_spec
    original_working = BaseAPI.working

    def record_run(run_dir, **fields):
        nonlocal current_model
        root = Path(run_dir).resolve()
        model_name = fields['model_name']
        # Toolkit run IDs have one-second precision. Reject collisions rather
        # than allowing a previous run's successful entries to survive upserts.
        if model_name not in run_dirs and (root / 'status.json').exists():
            raise RuntimeError(f'Evaluation run directory already contains status.json: {root}')
        result = original_status(run_dir, **fields)
        run_dirs[model_name] = root
        current_model = model_name
        return result

    def record_dataset(spec, *positional, **kwargs):
        nonlocal current_dataset
        dataset = original_build(spec, *positional, **kwargs)
        current_dataset = spec.dataset_alias_name
        if run.RANK == 0 and current_model is not None and dataset is not None:
            expected_indices[current_model, spec.dataset_alias_name] = list(dataset.data['index'])
        return dataset

    def record_working(judge):
        # After a failed check, datasets score by exact matching but still label results with the judge.
        # Copied judge files would hide the fallback again, so re-judging must reuse predictions only.
        working = original_working(judge)
        if not working:
            judge_fallbacks.append(f'{current_model} / {current_dataset}: judge {args.judge!r} failed its health check '
                                   'and was replaced by exact matching; re-judge with --mode eval --reuse-aux infer')
        return working

    run.upsert_run_status = record_run
    run.build_dataset_from_spec = record_dataset
    BaseAPI.working = record_working
    try:
        if args.api_mode:
            run.run_api_mode(args)
        else:
            run.run_local_mode(args)

        # Local mode has already destroyed its process group on normal return.
        # Rank 0 owns the status files; its failure makes torchrun fail the job.
        if run.RANK == 0:
            models = [args.model] if isinstance(args.model, str) else args.model
            failures = []
            if not models or not args.data:
                failures.append('No models or datasets were evaluated')
            for model_name in models or []:
                if args.api_mode:
                    model_name = model_name.replace('/', '--')
                root = run_dirs.get(model_name)
                if root is None:
                    failures.append(f'{model_name}: missing run status for this invocation')
                    continue
                failures.extend(benchmark_failures(run, root, model_name, args.data or [], expected_indices, args.mode))
            failures.extend(judge_fallbacks)
            if failures:
                raise RuntimeError('Evaluation failed:\n' + '\n'.join(f'- {failure}' for failure in failures))
    finally:
        run.upsert_run_status = original_status
        run.build_dataset_from_spec = original_build
        BaseAPI.working = original_working
        if distributed.is_initialized():
            distributed.destroy_process_group()


def main():
    # Let VLMEvalKit assign GPUs before importing Torch or the model.
    from VLMEvalKit import run
    import torch.distributed as distributed
    from main import reproduce
    from vlmeval.api import OpenAIWrapper
    from vlmeval_adapter import IPMoEVLMEval
    from vlmeval_baselines import BASELINES

    seed = int(os.environ['PYTHONHASHSEED'])
    reproduce(seed)
    checkpoint_path = os.environ.get('IPMOE_CHECKPOINT', f'checkpoint/seed_{seed}/checkpoint.pt')
    checkpoint_path = os.path.abspath(checkpoint_path)
    run.supported_VLM['IPMoE'] = partial(IPMoEVLMEval, checkpoint_path = checkpoint_path)
    run.supported_VLM.update(BASELINES)

    use_completion_tokens(OpenAIWrapper)
    # OpenAIWrapper logs the judge's API key at INFO, which would reach the console and run logs.
    logging.getLogger('vlmeval.api.gpt').setLevel(logging.WARNING)

    run.load_env()
    run_checked(run, run.parse_args(), distributed)


if __name__ == '__main__':
    main()
