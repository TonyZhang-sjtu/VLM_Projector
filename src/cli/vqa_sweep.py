"""VQA-only train -> best checkpoint -> evaluation launcher. One process, sequential ratios."""
import argparse
import codecs
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

from omegaconf import OmegaConf

from src.config import ROOT, file_hash, load_config, save_config
from src.data.neuroqa_manifest import write_json
from src.data.neuroqa_preflight import inspect_data, training_scope, compact_report, save_data_report, ensure_same_selection
from src.models.projectors.budget import resolve_budget

METHODS = ('avgpool3d', 'concat3d', 'convpool3d', 'mixer3d', 'local_packer3d',
           'attention_pool', 'perceiver_nv', 'perceiver_flamingo', 'adaptive_merge3d',
           'token_select3d', 'bindformer3d', 'mlp2x_gelu')
VERSIONS = {'v1': ('mixed.yaml', 'neuroqa_mni1mm'), 'v2': ('mixed_v2.yaml', 'neuroqa_mni1mm_v2')}


def parser():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--projector', choices=METHODS, default='avgpool3d')
    cli.add_argument('--ratios', nargs='+', type=int, default=[128, 256])
    cli.add_argument('--preprocessing-version', choices=VERSIONS, default='v1')
    cli.add_argument('--invalid-sample-policy', choices=('skip', 'strict', 'include_unreviewed'),
                     help='include_unreviewed retains pending v2 QC warnings and skips invalid cases')
    cli.add_argument('--config', action='append', default=[], help='Optional overlays; --set takes precedence')
    cli.add_argument('--set', action='append', default=[], dest='overrides')
    cli.add_argument('--seed', type=int, default=42)
    cli.add_argument('--concat-device', default='cpu')
    cli.add_argument('--run-tag')
    cli.add_argument('--output-root', type=Path, default=ROOT / 'runs/vqa/neuroqa')
    cli.add_argument('--results-root', type=Path, default=ROOT / 'results/vqa/neuroqa')
    cli.add_argument('--evaluate-test', action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument('--eval-split', choices=('train', 'val', 'test'), default='test')
    cli.add_argument('--eval-controls', nargs='+', choices=('normal', 'text', 'zero', 'shuffle'), default=['normal'])
    cli.add_argument('--eval-limit', type=int)
    cli.add_argument('--eval-ids-file', type=Path)
    cli.add_argument('--bootstrap', type=int, default=1000)
    cli.add_argument('--resume-run', type=Path, help='Directory containing launch.json; restores saved recipe and evaluation')
    cli.add_argument('--evaluate-only', action='store_true', help='Requires --resume-run; uses saved best.pt')
    cli.add_argument('--eval-tag', help='New evaluation directory suffix for an explicit retry')
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print config and commands; no data reads or output writes')
    mode.add_argument('--check-data', action='store_true', help='Read-only readiness check; no GPU/model allocation')
    return cli


def component(value):
    if not value or Path(value).name != value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError('Run/evaluation tags must be single directory names')
    return value


def absolute(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate_placement(config, visible=None):
    if config.training.device != 'cuda:0' or config.training.distributed != 'none':
        raise ValueError('VQA shells use one process with training.device=cuda:0')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '') if visible is None else visible
    ids = visible.split(',')
    if not all(i.isdigit() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to distinct physical GPU indices')
    other = config.training.projector_device
    expected = 2 if other == 'cuda:1' else 1
    if len(ids) != expected:
        raise ValueError(f'This placement requires {expected} visible GPU(s); got {visible}')
    if other not in (None, 'cpu', 'cuda:0', 'cuda:1') or (other in ('cpu', 'cuda:1') and config.projector.name != 'concat3d'):
        raise ValueError('CPU/second-GPU projector placement is supported for Concat only')


def build_launches(args):
    if args.resume_run:
        run = absolute(args.resume_run)
        spec = json.loads((run / 'launch.json').read_text())
        if spec['format_version'] != 1 or absolute(spec['run']) != run:
            raise ValueError('Invalid or moved VQA launch directory')
        config = load_config([run / 'launch_config.yaml'])
        if config.task.name != 'neuroqa':
            raise ValueError('Not a VQA run')
        if args.eval_tag:
            spec['evaluation']['tag'] = component(args.eval_tag)
        if args.evaluate_only:
            spec['evaluation']['enabled'] = True
        return [(config, spec)]
    if args.evaluate_only:
        raise ValueError('--evaluate-only requires --resume-run')
    if len(args.ratios) != len(set(args.ratios)):
        raise ValueError('Duplicate ratios in sweep')
    if (args.eval_limit is not None and args.eval_limit < 1) or args.bootstrap < 0:
        raise ValueError('Evaluation limit must be positive and bootstrap nonnegative')
    tag = component(args.run_tag or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S_') + uuid4().hex[:6])
    filename, adapter = VERSIONS[args.preprocessing_version]
    launches = []
    for ratio in args.ratios:
        overrides = [f'projector.name={args.projector}', f'projector.target_compression_ratio={ratio}', f'seed={args.seed}']
        if args.projector == 'concat3d':
            overrides.append(f'training.projector_device={args.concat_device}')
        if args.projector == 'bindformer3d':
            overrides.append('projector.params.knowledge_mode=none')
        config = load_config([ROOT / 'configs/vqa/neuroqa' / filename, *map(absolute, args.config)], overrides + args.overrides)
        if args.invalid_sample_policy:
            config.data.invalid_sample_policy = args.invalid_sample_policy
        if config.task.name != 'neuroqa' or config.preprocessing.name != adapter:
            raise ValueError('Selected preprocessing version/task disagrees with config overlays')
        if config.projector.name != args.projector or config.projector.target_compression_ratio != ratio:
            raise ValueError('Use --projector/--ratios to change the sweep identity')
        if config.training.resume:
            raise ValueError('Use --resume-run; do not inject training.resume into a new experiment')
        resolve_budget(config.projector, [49, 15, 13], [196, 240, 208])
        relative = (Path(component(config.data.target_version)) / args.preprocessing_version /
                    training_scope(config) / 'mixed' / config.projector.name / f'r{ratio}' / f's{config.seed}' / tag)
        run = absolute(args.output_root) / relative
        config.training.run_dir = str(run / 'train')
        spec = {'format_version': 1, 'run': str(run), 'preprocessing_version': args.preprocessing_version,
                'config': OmegaConf.to_container(config, resolve=True),
                'evaluation': {'enabled': args.evaluate_test, 'split': args.eval_split,
                    'controls': list(dict.fromkeys(args.eval_controls)), 'limit': args.eval_limit,
                    'ids_file': str(absolute(args.eval_ids_file)) if args.eval_ids_file else None,
                    'bootstrap': args.bootstrap, 'root': str(absolute(args.results_root) / relative),
                    'tag': component(args.eval_tag or args.eval_split)}}
        launches.append((config, spec))
    return launches


def commands(config, spec, resume=False):
    run = Path(spec['run'])
    train = [sys.executable, '-u', '-m', 'src.cli.vqa_train', '--config', str(run / 'launch_config.yaml')]
    train += ['--expected-data-selection-file', str(run / 'data_preflight.json')]
    if resume:
        train += ['--set', f'training.resume={run / "train/checkpoints/last.pt"}']
    evaluation = spec['evaluation']
    evaluations = []
    if evaluation['enabled']:
        for control in evaluation['controls']:
            output = Path(evaluation['root']) / evaluation['tag'] / control
            command = [sys.executable, '-u', '-m', 'src.cli.vqa_evaluate',
                       '--checkpoint', str(run / 'train/checkpoints/best.pt'), '--output', str(output),
                       '--split', evaluation['split'], '--control', control, '--bootstrap', str(evaluation['bootstrap'])]
            command += ['--expected-data-selection-file', str(run / 'data_preflight.json')]
            if evaluation['limit'] is not None:
                command += ['--limit', str(evaluation['limit'])]
            if evaluation['ids_file']:
                command += ['--ids-file', evaluation['ids_file']]
            evaluations.append((output, command))
    return train, evaluations


def run_logged(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8', newline='') as handle:
        handle.write('\n$ ' + shlex.join(command) + '\n')
        handle.flush()
        # Preserve tqdm carriage returns; text line iteration converts them to
        # newlines and would turn each progress refresh into another log line.
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            try:
                while True:
                    chunk = process.stdout.read1(4096)
                    value = decoder.decode(chunk, final=not chunk)
                    sys.stdout.write(value)
                    sys.stdout.flush()
                    handle.write(value)
                    handle.flush()
                    if not chunk:
                        break
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def execute(config, spec, args, readiness):
    run = Path(spec['run'])
    if not args.resume_run:
        run.mkdir(parents=True, exist_ok=False)
        save_config(config, run / 'launch_config.yaml')
        write_json(run / 'launch.json', spec)
    if args.resume_run and (run / 'data_preflight.json').is_file():
        previous = json.loads((run / 'data_preflight.json').read_text())
        ensure_same_selection(previous, readiness)
        ensure_same_selection(previous, readiness, 'evaluation_selection_fingerprint')
    save_data_report(run, readiness)
    train, evaluations = commands(config, spec, resume=bool(args.resume_run))
    status = {'status': 'running', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'preprocessing_version': spec['preprocessing_version'], 'scope': readiness['scope']}
    write_json(run / 'launcher_status.json', status)
    try:
        if not args.evaluate_only and not (run / 'train/result.json').exists():
            if args.resume_run and not (run / 'train/checkpoints/last.pt').is_file():
                raise FileNotFoundError('No last.pt to resume; choose a new run-tag for a fresh run')
            run_logged(train, run / 'logs/train.log')
        if evaluations:
            checkpoint = run / 'train/checkpoints/best.pt'
            sha = file_hash(checkpoint)
            for output, command in evaluations:
                signature = {'checkpoint_sha256': sha, 'command': command}
                complete = output / 'evaluation_complete.json'
                if complete.is_file() and json.loads(complete.read_text()) == signature and (output / 'metrics.json').is_file():
                    print(f'Already evaluated: {output}', flush=True)
                    continue
                if output.exists() and any(output.iterdir()):
                    raise FileExistsError(f'Incomplete/different evaluation at {output}; use --eval-tag NEW_NAME')
                run_logged(command, run / 'logs' / f'eval_{spec["evaluation"]["tag"]}_{output.name}.log')
                write_json(complete, signature)
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(run / 'launcher_status.json', status)


def main(argv=None):
    args = parser().parse_args(argv)
    launches = build_launches(args)
    if args.resume_run:
        print('Restoring saved run recipe/evaluation; fresh-run shell defaults and --set are ignored.', flush=True)
    for config, spec in launches:
        validate_placement(config)
        budget = config.projector.resolved_budget
        print(json.dumps({'run': spec['run'], 'preprocessing': spec['preprocessing_version'],
                          'cache': config.preprocessing.cache_dir, 'manifest': config.data.manifest_dir,
                          'invalid_sample_policy': config.data.get('invalid_sample_policy', 'strict'),
                          'modality_sampling': config.data.get('modality_sampling', 'legacy_exact'),
                          'N': budget.input_tokens, 'M': budget.expected_output_tokens,
                          'actual_ratio': budget.actual_ratio, 'scope': training_scope(config)}, indent=2), flush=True)
        if args.dry_run:
            print(OmegaConf.to_yaml({'projector': OmegaConf.to_container(config.projector, resolve=True),
                                    'training': OmegaConf.to_container(config.training, resolve=True),
                                    'generation': OmegaConf.to_container(config.generation, resolve=True)}), flush=True)
        train, evaluations = commands(config, spec, resume=bool(args.resume_run))
        if not args.evaluate_only:
            print(shlex.join(train), flush=True)
        for _, command in evaluations:
            print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    # All ratios share data; check once, before creating run dirs or loading GPUs.
    config, spec = launches[0]
    readiness = inspect_data(config, spec['evaluation'] if spec['evaluation']['enabled'] else None)
    print(json.dumps(compact_report(readiness), indent=2), flush=True)
    if readiness['status'] != 'passed':
        return 2
    if args.check_data:
        return 0
    for config, spec in launches:
        per_run = copy.deepcopy(readiness)
        if 'geometry' in per_run:
            per_run['geometry']['budget'] = OmegaConf.to_container(config.projector.resolved_budget, resolve=True)
        execute(config, spec, args, per_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
