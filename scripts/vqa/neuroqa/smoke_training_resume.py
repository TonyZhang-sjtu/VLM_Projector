#!/usr/bin/env python3
"""Exercise the production QA trainer, all schedule phases, validation and resume."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from src.config import load_config
from src.data.neuroqa_manifest import write_json
from src.training.vqa_trainer import VQATrainer


class PlannedInterruption(Exception):
    pass


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--config', default='configs/vqa/neuroqa/mixed.yaml')
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selections = args.output / 'selections'
    write_json(selections / 'train.json', ['BGLI-00001','BGLI-00873','BMEN-00001','BMEN-00365'])
    write_json(selections / 'val.json', [f'{q}__{m}' for q in ('BGLI-00121','BGLI-00551','BGLI-00058') for m in ('T1w','T2w')])
    run = args.output / 'run'
    overrides = [f'training.run_dir={run.resolve()}',
        f'training.train_ids_file={(selections / "train.json").resolve()}',
        f'training.val_ids_file={(selections / "val.json").resolve()}',
        'training.effective_batch_size=2','training.max_epochs=3',
        'training.projector_warmup_steps=2','training.joint_warmup_steps=2',
        'training.save_every_steps=1','training.early_stopping_enabled=false']
    config = load_config([ROOT / args.config], overrides)
    trainer = VQATrainer(config)
    original_save = trainer.save

    def stop_after_save(name):
        original_save(name)
        if trainer.state['global_step'] == 2:
            raise PlannedInterruption('Exercise resume at the projector-to-LoRA boundary')

    trainer.save = stop_after_save
    try:
        trainer.train()
        raise AssertionError('Expected controlled interruption')
    except PlannedInterruption:
        pass
    # Remove the bound-method references before loading another real 8B model.
    trainer.save = None
    del original_save, trainer
    gc.collect()
    torch.cuda.empty_cache()
    config.training.resume = str((run / 'checkpoints/last.pt').resolve())
    resumed = VQATrainer(config)
    assert resumed.state['global_step'] == 2
    final = resumed.train()
    steps = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
    assert [r['step'] for r in steps] == [1,2,3,4,5,6]
    assert final['modality_counts'] == {'T1W':6,'T2W':6}
    assert {r['phase_used'] for r in steps} == {'projector_warmup','joint_warmup','joint_cosine'}
    assert (run / 'checkpoints/best.pt').is_file()
    assert len(list((run / 'validation').glob('*/metrics.json'))) == 3
    write_json(args.output / 'result.json', {'status':'passed','resume_step':2,'steps':6,
        'modalities':final['modality_counts'],'all_three_schedule_phases':True,
        'production_trainer':True,'validation_formats':['YN','MCQ','Open'],
        'full_validation_subset_per_epoch':6,'scope':'engineering_smoke_not_benchmark'})


if __name__ == '__main__':
    main()
