import json
from pathlib import Path

from src.cli.common import parser, config_from
from src.data.neuroqa_preflight import inspect_data, compact_report
from src.training.vqa_trainer import VQATrainer


def main():
    cli = parser('Train the single-volume NeuroQA benchmark')
    cli.add_argument('--dry-run', action='store_true')
    cli.add_argument('--invalid-sample-policy', choices=('skip', 'strict', 'include_unreviewed'))
    cli.add_argument('--expected-data-selection-file', type=Path)
    args = cli.parse_args()
    config = config_from(args)
    if args.invalid_sample_policy:
        config.data.invalid_sample_policy = args.invalid_sample_policy
    if config.task.name != 'neuroqa':
        cli.error('Use --config configs/vqa/neuroqa/mixed.yaml')
    if args.dry_run:
        report = inspect_data(config)
        print(json.dumps(compact_report(report), indent=2))
        if report['status'] != 'passed':
            raise SystemExit(2)
        return
    expected = json.loads(args.expected_data_selection_file.read_text()) if args.expected_data_selection_file else None
    trainer = VQATrainer(config, expected_selection=expected)
    trainer.train()


if __name__ == '__main__':
    main()
