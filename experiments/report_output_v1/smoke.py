"""Bounded real-model R1/R2 checks. No sweep or formal experiment is launched."""
import argparse
import json
from pathlib import Path
import time


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from .runtime import install
    install()
    import torch
    from omegaconf import OmegaConf
    from src.config import path_of
    from src.data.manifest import load_manifests
    from src.data.dataset import VolumeDataset
    from src.preprocessing.registry import build_preprocessor
    from src.models.multimodal import MultimodalModel
    from src.training.trainer import seed_all
    from src.training.optim import build_optimizer, build_joint_scheduler
    from src.training.schedule import resolve_joint_schedule
    from src.training.checkpoint import save_checkpoint, load_checkpoint
    from src.training.loss import combined_loss
    from .store import OutputStore
    c = OmegaConf.create(json.loads(args.plan.read_text())['runs'][0]['config'])
    c.projector.params.report_output.span_weight = .1
    c.projector.params.report_output.group_weight = .1
    c.projector.params.report_output.warmup_steps = 0
    c.training.run_dir = str(args.output)
    c.generation.max_new_tokens = 8  # Explicit smoke-only generation budget.
    torch.cuda.set_device(0)
    seed_all(c.seed)
    store = OutputStore(path_of(c, c.projector.params.report_output.path))
    samples = load_manifests(c)[0]
    lookup = {s.id: s for s in samples['train']}
    first = next(rid for rid, r in store.rows['train'].items()
                 if sum(len(g['candidates']) for g in r['groups']) >= 3 and len(r['facts']) >= 3)
    partner = next(ca['partner_id'] for g in store.rows['train'][first]['groups'] for ca in g['candidates'])
    selected = [first, partner]
    preprocessor = build_preprocessor(c)
    data = VolumeDataset([lookup[rid] for rid in selected], preprocessor)
    items = [data[i] for i in range(len(data))]
    model = MultimodalModel(c).to('cuda:0')
    model.train()
    optimizer = build_optimizer(model, c)
    windows = (len(samples['train'])+c.training.effective_batch_size-1)//c.training.effective_batch_size
    plan = resolve_joint_schedule(c.training, windows*c.training.max_epochs)
    scheduler = build_joint_scheduler(optimizer, plan)
    step = plan.cosine_start
    # Advance only the SMOKE scheduler to test the joint vision-LoRA phase.
    for _ in range(step):
        optimizer.step(); scheduler.step()
    model.encoder.set_phase(plan.phase(step))
    records = []
    for mode, group_weight in [('R1', 0.), ('R2', .1)]:
        model.output_settings['group_weight'] = group_weight
        model.set_supervision_step(step)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        tokens = sum(model.decoder.answer_count(i['sample']) for i in items)
        totals = {}
        for item in items:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model([item])
                loss = combined_loss(result, tokens, sample_count=len(items))
            if not torch.isfinite(loss):
                raise FloatingPointError('Smoke loss is nonfinite')
            loss.backward()
            expected = c.projector.resolved_budget.expected_output_tokens
            assert result['projector_token_counts'] == [expected]
            for key, value in result['auxiliary_metrics'].items():
                totals[key] = totals.get(key, 0.)+value
            del loss, result
        gradients = {name: sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None)
                     for name, module in [('encoder', model.encoder), ('projector', model.projector), ('decoder', model.decoder)]}
        assert gradients['encoder'] > 0 and gradients['projector'] > 0 and gradients['decoder'] == 0
        expected_pairs = sum(len(g['candidates']) for rid in selected for g in store.rows['train'][rid]['groups']) if group_weight else 0
        assert totals['pair_count'] == expected_pairs
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step(); step += 1
        torch.cuda.synchronize()
        record = dict(mode=mode, metrics=totals, gradient_norms=gradients,
                      global_grad_norm=float(norm), seconds=time.perf_counter()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        records.append(record)
        print(json.dumps(record), flush=True)
    optimizer.zero_grad(set_to_none=True)
    state = dict(global_step=step, phase=plan.phase(step), schedule=plan.to_dict(), epoch=0,
                 next_window=2, world_size=1)
    checkpoint = args.output/'smoke.pt'
    proof = {'contract_id': 'isolated_smoke'}
    save_checkpoint(checkpoint, model, optimizer, scheduler, state, proof)
    _, parameter = next(iter(model.projector.named_parameters()))
    before = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1)
    load_checkpoint(checkpoint, model, 'resume', optimizer, scheduler, proof)
    assert torch.equal(before, parameter)
    assert scheduler.last_epoch == step
    # Exercise the actual opt-in epoch validation wrapper and RNG restoration.
    val_lookup = {s.id: s for s in samples['val']}
    val_id = next(rid for rid, row in store.rows['val'].items() if len(row['facts']) >= 2)
    val_data = VolumeDataset([val_lookup[val_id]], preprocessor)
    model._output_training_owner = True
    model._output_validation_calls = 0
    model.set_supervision_step(step-1)
    model.train()
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    import src.training.trainer as training
    metrics = training.validate(model, val_data)
    assert 'report_facts_generation' in metrics and model.training
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    # Exercise the real Trainer.save override without allocating a second 8B LM.
    owner = object.__new__(training.Trainer)
    owner.model = model; owner.optimizer = optimizer; owner.scheduler = scheduler
    owner.state = state; owner.provenance = proof; owner.output = args.output
    owner.rank = 0; owner.world = 1
    owner.save('last.pt')
    assert (args.output/'checkpoints/best_facts.pt').exists()
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = model([items[0]])
        assert 'auxiliary_loss_sum' not in result
    summary = dict(status='passed', ratio=int(c.projector.target_compression_ratio), sample_ids=selected,
                   records=records, checkpoint_resume=True, validation_generation=metrics,
                   validation_preserves_rng=True, best_facts_saved=True,
                   evaluation_has_no_auxiliary_loss=True,
                   scope='isolated_two_sample_smoke_not_a_quality_experiment')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'ratio': summary['ratio'], 'summary': str(args.output/'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
