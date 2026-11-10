"""CPU-only end-to-end RadGraph-XL extraction on the frozen synthetic probes.

Uses the existing weights read-only. No training cache/config is rewritten.
These are author-defined probes, not clinician-validated accuracy estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def expected_facts():
    def f(side, site, status='present'):
        return dict(side=side, site=site, status=status)
    lf = f('left', 'frontal_lobe'); rt = f('right', 'temporal_lobe')
    pairs = [([lf], [lf]) for _ in range(28)]
    pairs[3] = ([lf], [f(None, 'frontal_lobe')])
    pairs[4] = ([lf], [f('right', 'frontal_lobe')])
    pairs[5] = ([lf], [f('left', 'temporal_lobe')])
    pairs[6] = ([lf], [f('left', 'frontal_lobe', 'absent')])
    pairs[7] = ([lf], [f('left', 'frontal_lobe', 'uncertain')])
    pairs[12] = ([lf], [f('bilateral', 'frontal_lobe')])
    pairs[13] = ([f('bilateral', 'frontal_lobe')], [lf, f('right', 'frontal_lobe')])
    pairs[14] = ([lf, rt], [rt, lf])
    pairs[15] = ([lf, rt], [f('right', 'frontal_lobe'), f('left', 'temporal_lobe')])
    pairs[16] = ([lf, rt], [lf])
    pairs[17] = ([lf], [lf, rt])
    pairs[18] = ([lf, f('left', 'parietal_lobe')], [lf, f('left', 'parietal_lobe')])
    pairs[19] = ([f(None, 'brainstem')], [f(None, 'brainstem')])
    pairs[20] = ([f('left', 'insula')], [f('left', 'insula')])
    pairs[21] = ([f('midline', 'corpus_callosum')], [f('midline', 'corpus_callosum')])
    pairs[23] = ([f('midline', None)], [f('bilateral', None)])
    pairs[24] = ([], [lf])
    pairs[26] = ([lf], [])
    pairs[27] = ([f(None, 'pons')], [f(None, 'brainstem')])
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--reuse-graphs', type=Path, help='Reuse frozen raw graphs without loading a model')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from .prepare import stress_cases
    from .protocol import label_sets, score_sets
    if args.protocol == 'v2':
        from .protocol_v2 import extract_graph
    else:
        from .protocol import extract_graph
    assets = ROOT/'assets/models/evaluation'
    config = assets/'radgraph/runtime/radgraph-xl/config.json'
    before = hashlib.sha256(config.read_bytes()).hexdigest()
    if args.reuse_graphs:
        frozen = {r['id']:r for r in map(json.loads,args.reuse_graphs.read_text().splitlines())}
    else:
        sys.path.insert(0, str(ROOT/'third_party/evaluation/python_deps'))
        import torch
        from radgraph import F1RadGraph
        torch.set_num_threads(2)
        model = F1RadGraph(model_type='radgraph-xl', reward_level='all', cuda=-1,
                          model_cache_dir=str(assets/'radgraph/runtime'),
                          tokenizer_cache_dir=str(assets/'hub'))
    rows = []
    with (args.output/'cases.jsonl').open('w') as stream:
        for case, (rf, pf) in zip(stress_cases(), expected_facts()):
            if args.reuse_graphs:
                old = frozen[case['id']]
                assert old['reference'] == case['reference'] and old['prediction'] == case['prediction']
                preds, refs = [old['prediction_graph']], [old['reference_graph']]
            else:
                with torch.inference_mode():
                    _, _, preds, refs = model(refs=[case['reference']], hyps=[case['prediction']])
            assert len(preds) == len(refs) == 1
            ref, pred = extract_graph(refs[0]), extract_graph(preds[0])
            expected = score_sets(dict(facts=rf), dict(facts=pf))
            actual = score_sets(ref, pred)
            agrees = {}
            for key in ['side', 'site', 'joint']:
                a, e = actual[key], expected[key]
                agrees[key] = (a['eligible'] == e['eligible'] and
                               (not e['eligible'] or (a['tp'], a['fp'], a['fn']) == (e['tp'], e['fp'], e['fn'])))
            def same_labels(auto, facts):
                a, e = label_sets(auto), label_sets(dict(facts=facts))
                return {k:a[k] == e[k] for k in ['side','site','joint','eligible']}
            row = dict(**case, reference_graph=refs[0], prediction_graph=preds[0],
                       reference_extraction=ref, prediction_extraction=pred,
                       expected_reference_facts=rf, expected_prediction_facts=pf,
                       expected_scores=expected, actual_scores=actual, score_agrees=agrees,
                       reference_labels_agree=same_labels(ref,rf), prediction_labels_agree=same_labels(pred,pf))
            rows.append(row)
            stream.write(json.dumps(row)+'\n'); stream.flush()
            print(case['id'], agrees, flush=True)
    assert hashlib.sha256(config.read_bytes()).hexdigest() == before
    summary = dict(n_probes=len(rows), source='author_defined_synthetic_probes',
                   clinical_validation=False, device='cpu', threads=2, protocol=args.protocol,
                   raw_graphs_reused=bool(args.reuse_graphs),
                   reference_config_unchanged=True,
                   full_score_agreement=sum(all(r['score_agrees'].values()) for r in rows),
                   score_agreement_by_dimension={k:sum(r['score_agrees'][k] for r in rows) for k in ['side','site','joint']},
                   discrepant_ids=[r['id'] for r in rows if not all(r['score_agrees'].values())])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
