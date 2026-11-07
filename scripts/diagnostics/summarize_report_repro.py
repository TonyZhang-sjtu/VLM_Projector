"""Summarize the GPU reproducibility probes using CPU tensor comparisons."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(2)


def read(path):
    return json.loads(path.read_text())


def tensor_difference(left, right):
    a = torch.load(left, weights_only=True, map_location='cpu', mmap=True)
    b = torch.load(right, weights_only=True, map_location='cpu', mmap=True)
    assert a.keys() == b.keys()
    changed = []
    for key in a:
        if not torch.equal(a[key], b[key]):
            delta = a[key].float()-b[key].float()
            changed.append({'name':key, 'max_abs':float(delta.abs().max()),
                            'different_values':int(torch.count_nonzero(delta))})
    return {'all_equal': not changed, 'changed_tensors':len(changed),
            'changed_values':sum(x['different_values'] for x in changed),
            'max_abs':max((x['max_abs'] for x in changed),default=0.), 'tensors':changed}


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('run_root', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'run_root':str(args.run_root.resolve()),'comparisons':[], 'seed_changes':[],
               'historical_comparisons':[], 'trace_comparisons':[],
               'limits':['Short training only; no complete retraining or test RadGraph evaluation.',
                         'Observed equality in two trials does not prove universal bitwise reproducibility.',
                         'Encoder-stage trace divergence does not identify the individual kernel responsible.']}
    table = []
    for method in ['perceiver128','localread256']:
        folder = args.run_root/method
        for mode in ['fresh42','joint42']:
            a,b = [read(folder/(mode+'_'+label)/'probe.json') for label in ['a','b']]
            assert a['status'] == b['status'] == 'complete'
            delta = tensor_difference(folder/(mode+'_a')/'diagnostic_trainable_tensors.pt',
                                      folder/(mode+'_b')/'diagnostic_trainable_tensors.pt')
            row = {'method':method,'mode':mode,'steps_per_trial':a['completed_steps'],
                   'step_ids':[a['initial_step']+i+1 for i in range(a['completed_steps'])],
                   'initial_weights_equal':a['initial_weights']==b['initial_weights'],
                   'initial_rng_equal':all(a[k]==b[k] for k in ['initial_cuda_rng_sha256','initial_cpu_rng_sha256']),
                   'losses_a':a['losses'],'losses_b':b['losses'],
                   'max_loss_difference':max(abs(x-y) for x,y in zip(a['losses'],b['losses'])),
                   'weight_difference':delta,'peak_allocated_gib':max(a['peak_allocated_gib'],b['peak_allocated_gib'])}
            summary['comparisons'].append(row)
            table.append({k:row[k] for k in ['method','mode','steps_per_trial','initial_weights_equal','max_loss_difference','peak_allocated_gib']} | {
                'final_weights_equal':delta['all_equal'], 'weight_max_abs_difference':delta['max_abs']})
        a,b = [read(folder/name/'probe.json') for name in ['fresh42_a','fresh43']]
        summary['seed_changes'].append({'method':method,'seeds':[42,43],
            'initial_weights_equal':a['initial_weights']==b['initial_weights'],
            'losses_seed42':a['losses'],'losses_seed43':b['losses']})
        old = Path(a['source'])/'joint_lora'
        records = [json.loads(x) for x in (old/'train.jsonl').read_text().splitlines()][:3]
        proof = read(old/'provenance.json')
        current = read(folder/'fresh42_a/provenance.json')
        summary['historical_comparisons'].append({'method':method,'source':str(old),
            'contract_equal':proof['contract_id']==current['contract_id'],
            'changed_source_files':[k for k,v in proof['source_files'].items() if current['source_files'].get(k)!=v],
            'historical_losses':[r['answer_ce'] for r in records],'current_losses':a['losses'],
            'source_metrics':read(Path(a['source'])/'evaluation/best_test/metric.json')['overall']})
    folder = args.run_root/'localread256'
    ref = read(folder/'trace42_a/first_forward.json')
    for name in ['trace42_b','trace42_c']:
        other = read(folder/name/'first_forward.json')
        diff = tensor_difference(folder/'trace42_a/first_forward_tensors.pt',folder/name/'first_forward_tensors.pt')
        summary['trace_comparisons'].append({'comparison':'trace42_a vs '+name,
            'input_metadata_equal':ref['metadata']==other['metadata'],
            'first_differing_recorded_stage':next((k for k,v in ref['tensor_hashes'].items() if other['tensor_hashes'][k]!=v),None),
            'tensor_difference':diff})
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (args.output/'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(table,indent=2))


if __name__ == '__main__':
    main()
