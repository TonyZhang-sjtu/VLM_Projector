"""N-series: train once, then audit and evaluate named epoch/best checkpoints."""
import csv
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.cli import baseline_sweep as sweep
from src.config import digest, file_hash, save_config
from src.evaluation.ft_diagnostics import summarize_evaluation


def evaluation_jobs(run, split):
    jobs = []
    epochs = run['config']['training']['checkpoint_epochs']
    for epoch in [*epochs, None]:
        label = f'epoch_{epoch:02d}' if epoch is not None else 'best'
        checkpoint = Path(run['checkpoint']).with_name(label + '.pt')
        directory = Path(run['directory']) / 'evaluation' / f'{label}_{split}'
        command = list(run['eval_command'])
        command[command.index('--checkpoint') + 1] = str(checkpoint)
        command[command.index('--output-dir') + 1] = str(directory)
        jobs.append({'label': label, 'epoch': epoch, 'checkpoint': str(checkpoint),
                     'directory': str(directory), 'command': command})
    return jobs


def build_plan(args):
    plan = sweep.build_plan(args)
    sweep.validate_full_test_plan(plan)
    if plan['projector'] != 'avgpool3d' or plan['ratios'] != [8]:
        raise ValueError('N-series requires AvgPool3D at ratio 8')
    plan.update(runner='ft_experiments', require_full_test=True)
    for run in plan['runs']:
        training = run['config']['training']
        if not training['checkpoint_epochs']:
            raise ValueError('N-series requires training.checkpoint_epochs')
        if training['max_steps'] is not None:
            raise ValueError('N-series uses complete epochs; max_steps must be null')
        if not training['log_optimizer_diagnostics']:
            raise ValueError('N-series requires optimizer diagnostics')
        run['evaluations'] = evaluation_jobs(run, plan['eval_split'])
    return plan


def checkpoint_metadata(path, run, expected_epoch=None):
    """Validate the saved training identity; fingerprint all evaluation weights."""
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if saved.get('format_version') != 2 or saved.get('stage') != 'joint_lora':
        raise ValueError(f'Unexpected checkpoint stage/format: {path}')
    if digest(saved['config']) != digest(run['config']):
        raise ValueError(f'Checkpoint config differs from the experiment plan: {path}')
    state = saved['state']
    if state['schedule'] != run['schedule']:
        raise ValueError(f'Checkpoint schedule differs from the experiment plan: {path}')
    for split in ('train', 'val'):
        if state[f'{split}_selection'] != run['data_identity'][f'{split}_ids_sha256']:
            raise ValueError(f'Checkpoint {split} selection differs from the plan')
    epoch, step = state['epoch'], state['global_step']
    if state['next_window'] != 0 or step != epoch * run['updates_per_epoch']:
        raise ValueError('Requested checkpoint is not a completed epoch')
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError('Checkpoint epoch does not match its requested label')
    if not 1 <= epoch <= run['config']['training']['max_epochs']:
        raise ValueError('Checkpoint epoch is outside the training budget')
    # A repeated serialization can have a different file hash. Reuse scoring
    # only when config, step, provenance, and every projector/LoRA tensor agree.
    identity = hashlib.sha256(digest({
        'config': saved['config'], 'step': step,
        'contract': saved['provenance']['contract_id'],
    }).encode())
    for component in ('projector', 'encoder'):
        for name, tensor in sorted(saved[component].items()):
            identity.update(digest([component, name, str(tensor.dtype), list(tensor.shape)]).encode())
            identity.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return {'epoch': epoch, 'step': step, 'checkpoint_sha256': file_hash(path),
            'model_identity_sha256': identity.hexdigest(),
            'best_validation_ce_so_far': state['best_val_ce']}


def audit_result(result, run, metadata, directory):
    sweep.validate_full_test_result(result, run)
    if result.get('status') not in ('complete', 'partial'):
        raise ValueError('Evaluation did not complete')
    if result.get('checkpoint_sha256') != metadata['checkpoint_sha256']:
        raise ValueError('Evaluation/checkpoint SHA256 mismatch')
    if result.get('manifest_summary') != run['manifest_summary']:
        raise ValueError('Evaluation manifest identity differs from training preflight')
    ids = json.loads((Path(directory) / 'expected_ids.json').read_text())
    if digest(ids) != run['data_identity']['evaluation_ids_sha256']:
        raise ValueError('Evaluation IDs differ from the planned test split')


def write_summary(run, entries):
    output = Path(run['directory'])
    sweep.write_json(output / 'checkpoint_results.json', {'checkpoints': entries})
    fields = ['checkpoint_label', 'status', 'epoch', 'step', 'checkpoint_sha256',
              'evaluated_checkpoint_sha256', 'reused_from', 'metric_path',
              'bleu4', 'rougeL', 'green', 'radgraph_simple', 'radgraph_partial', 'radgraph_complete',
              'rate', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'answer_ce',
              'green_valid', 'green_failed', 'mean_generated_tokens', 'reason']
    temporary = output / 'checkpoint_results.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for entry in entries:
            diagnostics = entry.get('diagnostics', {})
            writer.writerow({**entry, **entry.get('overall', {}),
                             **{key: diagnostics.get(key) for key in
                                ('green_valid', 'green_failed', 'mean_generated_tokens')}})
    temporary.replace(output / 'checkpoint_results.csv')


def execute_plan(plan, resume=False):
    sweep.validate_full_test_plan(plan)
    if plan.get('runner') != 'ft_experiments' or len(plan['runs']) != 1:
        raise ValueError('Not an N-series experiment plan')
    run = plan['runs'][0]
    if run.get('evaluations') != evaluation_jobs(run, plan['eval_split']):
        raise ValueError('Saved checkpoint evaluation list differs from its config')
    root = Path(plan['run_root'])
    if not resume:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()  # Refuse collisions, including concurrent launches.
        sweep.write_json(root / 'sweep_plan.json', plan)
        save_config(OmegaConf.create(run['config']), run['config_path'])
    state = {'status': 'running', 'phase': 'training', 'active_checkpoint': None}
    entries, completed_models = [], {}
    try:
        sweep.write_json(root / 'sweep_status.json', state)
        best = Path(run['checkpoint'])
        if resume and sweep.training_finished(best):
            print('Reusing this experiment\'s completed training; resuming evaluation.', flush=True)
        else:
            if resume and (best.parent.parent / 'train.jsonl').exists():
                raise ValueError('Training was interrupted. Resume training explicitly from this run\'s last.pt first; '
                                 '--resume-run here resumes evaluation only')
            sweep.run_logged(run['train_command'], Path(run['directory']) / 'train_console.log', append=resume)
        if not sweep.training_finished(best):
            raise ValueError('Training completion and best.pt are required before test evaluation')
        training_state = json.loads((best.parent.parent / 'training_state.json').read_text())
        for job in run['evaluations']:
            state.update(phase='evaluation', active_checkpoint=job['label'])
            sweep.write_json(root / 'sweep_status.json', state)
            path, directory = Path(job['checkpoint']), Path(job['directory'])
            if not path.is_file():
                if (job['epoch'] is not None and training_state['stop_reason'] == 'early_stopping'
                        and training_state['epoch'] < job['epoch']):
                    entries.append({'checkpoint_label': job['label'], 'status': 'not_reached',
                                    'epoch': job['epoch'], 'reason': 'CE early stopping before requested epoch'})
                    write_summary(run, entries)
                    continue
                raise FileNotFoundError(f'Missing requested checkpoint: {path}')
            metadata = checkpoint_metadata(path, run, job['epoch'])
            entry = {'checkpoint_label': job['label'], 'checkpoint': str(path), **metadata}
            identity = metadata['model_identity_sha256']
            if identity in completed_models:
                source = completed_models[identity]
                # Keep the actual evaluated file identity explicit: alias.json
                # points to raw results; no copied/mislabeled metric.json.
                entry.update(status=source['status'], reused_from=source['checkpoint_label'],
                             evaluated_checkpoint_sha256=source['evaluated_checkpoint_sha256'],
                             metric_path=source['metric_path'], overall=source['overall'],
                             diagnostics=source['diagnostics'], partial_metrics=source['partial_metrics'])
                directory.mkdir(parents=True, exist_ok=True)
                sweep.write_json(directory / 'alias.json', entry)
            else:
                metric_path = directory / 'metric.json'
                if resume and metric_path.is_file():
                    result = json.loads(metric_path.read_text())
                else:
                    command = list(job['command'])
                    if resume and directory.exists():
                        command.append('--resume')
                    sweep.run_logged(command, Path(run['directory']) / f'evaluation_{job["label"]}_console.log',
                                     append=resume)
                    result = json.loads(metric_path.read_text())
                audit_result(result, run, metadata, directory)
                diagnostics = summarize_evaluation(directory, result, run['data_identity']['evaluation_ids_sha256'])
                entry.update(status=result['status'], evaluated_checkpoint_sha256=metadata['checkpoint_sha256'],
                             reused_from=None, metric_path=str(metric_path), overall=result['overall'],
                             diagnostics=diagnostics, partial_metrics=result.get('partial_metrics', {}))
                completed_models[identity] = entry
            entries.append(entry)
            write_summary(run, entries)
        state.update(status='partial' if any(e['status'] != 'complete' for e in entries) else 'complete',
                     phase='complete', active_checkpoint=None)
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        sweep.write_json(root / 'sweep_status.json', state)
    print(f'N-series experiment complete: {root}', flush=True)


def main():
    cli = sweep.parser()
    cli.description = __doc__
    args = cli.parse_args()
    try:
        plan = (sweep.load_resume_plan(args.resume_run, args.projector)
                if args.resume_run else build_plan(args))
        if plan.get('runner') != 'ft_experiments':
            raise ValueError('Use the matching launcher to resume this plan')
        sweep.validate_full_test_plan(plan)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        cli.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    execute_plan(plan, resume=args.resume_run is not None)


if __name__ == '__main__':
    main()
