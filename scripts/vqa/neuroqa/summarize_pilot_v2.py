#!/usr/bin/env python3
"""Summarize all frozen-pilot candidates and their preregistered safety checks."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import SimpleITK as sitk

from src.config import file_hash
from src.data.neuroqa_manifest import read_rows, write_json
from src.preprocessing.neuroqa_smri_v2 import issues
from src.preprocessing.neuroqa_smri import TEMPLATE, compose_pull, warp
from src.preprocessing.neuroqa_registration_v2 import prepare_native


# Preliminary engineering review gates, fixed before looking at all pilot outcomes.
# These are not clinical validity thresholds. Pilot evidence can require a revised
# protocol, but no val/test QA scores may be used to select it.
POLICY = {'min_singular': .5, 'max_singular': 1.4, 'min_dice': .8,
          'max_outside_fraction': .001, 'max_dice_stage_drop': .05,
          'min_t2_t1_dice': .9}


def main():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    root = ROOT/'results/vqa/neuroqa/preprocessing_v2_plan'
    pilot = json.loads((root/'pilot_manifest.json').read_text())
    template_path = ROOT/'assets/preprocessing/neuroqa_mni1mm_v1/templates'/TEMPLATE
    template_hash = file_hash(template_path)
    template = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    template_values = sitk.GetArrayFromImage(template)[::2, ::2, ::2]
    evaluation_points = template_values > 0
    fixed = template_values[evaluation_points]
    fixed_upper = float(np.percentile(fixed, 99.9))
    index = {r['case_id']:r for r in read_rows(ROOT/'assets/data/NeuroQA_derived/official_holdout_patient_guard_v1/source_index.jsonl')}
    results = {'scope':'train_pilot_only', 'pilot_manifest_sha256':file_hash(root/'pilot_manifest.json'),
               'preliminary_engineering_policy':POLICY, 'candidate_summaries':{}, 'cases':[]}
    for case_id in pilot['case_ids']:
        source = index[case_id]
        native = {m:prepare_native(sitk.ReadImage(p, sitk.sitkFloat32))[0] for m,p in source['images'].items()}
        source_hashes = {m:file_hash(path) for m,path in source['images'].items()}
        masks = {m:sitk.Cast(img != 0, sitk.sitkUInt8) for m,img in native.items()}
        a = sitk.GetArrayFromImage(masks['T1w']) > 0
        upper = {m:float(np.percentile(sitk.GetArrayFromImage(img)[sitk.GetArrayFromImage(masks[m]) > 0], 99.9)) for m,img in native.items()}

        def nmi(modality, transform):
            moved = sitk.GetArrayFromImage(warp(native[modality], template, transform))[::2,::2,::2][evaluation_points]
            histogram = np.histogram2d(np.clip(fixed, 0, fixed_upper), np.clip(moved, 0, upper[modality]),
                bins=64, range=[[0,fixed_upper],[0,upper[modality]]])[0]
            p = histogram/histogram.sum()
            def entropy(values):
                x = values[values > 0]
                return float(-np.sum(x*np.log(x)))
            return (entropy(p.sum(axis=0))+entropy(p.sum(axis=1)))/entropy(p)
        entry = {'case_id':case_id,'candidates':{}}
        for candidate in ('B','C','D'):
            folder = root/'pilot'/candidate/case_id
            p = folder/'result.json'
            if not p.exists():
                entry['candidates'][candidate] = {'status':'missing'}
                continue
            data = json.loads(p.read_text())
            if data['contract']['sources'] != source_hashes or data['contract']['template'] != template_hash:
                raise ValueError(f'Source changed during pilot: {case_id}')
            for name, expected in data['artifacts'].items():
                if file_hash(folder/name) != expected:
                    raise ValueError(f'Pilot artifact changed: {folder/name}')
            result_sha = file_hash(p)
            cache_path = root/'pilot_nmi_cache'/candidate/f'{case_id}.json'
            cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
            if cached.get('result_sha256') == result_sha and cached.get('algorithm') == 'nmi_fixed_template_stride2_v1':
                metrics = cached['metrics']
                entry['candidates'][candidate] = {'status':'measured','metrics':metrics,
                    'issues':issues(metrics,POLICY), 'seconds':data['seconds'], 'result_sha256':result_sha}
                continue
            t = sitk.ReadTransform(str(folder/'t2_to_t1.tfm'))
            b = sitk.GetArrayFromImage(warp(masks['T2w'], native['T1w'], t, True)) > 0
            metrics = {**data['metrics'], 'T2_to_T1_mask_dice':float(2*np.logical_and(a,b).sum()/max(1,a.sum()+b.sum()))}
            initial = sitk.ReadTransform(str(folder/'initial.tfm'))
            rigid = sitk.ReadTransform(str(folder/'rigid.tfm'))
            for m in ('T1w','T2w'):
                stage_rigid = rigid if m == 'T1w' else compose_pull(t, rigid)
                metrics[m]['nmi_fixed_template_points'] = {
                    'initial':nmi(m, initial), 'rigid':nmi(m, stage_rigid),
                    'final':nmi(m, sitk.ReadTransform(str(folder/f'{m}_final.tfm')))}
            write_json(cache_path, {'result_sha256':result_sha,
                'algorithm':'nmi_fixed_template_stride2_v1','metrics':metrics})
            entry['candidates'][candidate] = {'status':'measured','metrics':metrics,
                'issues':issues(metrics,POLICY), 'seconds':data['seconds'],
                'result_sha256':result_sha}
        prior = ROOT/'assets/preprocessing/neuroqa_mni1mm_v1/cases'/case_id
        p = prior/('complete.json' if (prior/'complete.json').exists() else 'qc.json')
        if p.exists():
            d = json.loads(p.read_text());q = d.get('qc',d)
            entry['v1_dice'] = {m:q[m]['template_foreground_dice'] for m in ('T1w','T2w')}
        results['cases'].append(entry)
    for candidate in ('B','C','D'):
        rows = [(x,x['candidates'][candidate]) for x in results['cases']]
        complete = [v for _,v in rows if v['status']=='measured']
        dice = [v['metrics'][m]['final']['dice'] for v in complete for m in ('T1w','T2w')]
        failures = [{'case_id':x['case_id'],'issues':v['issues']} for x,v in rows if v['status']=='measured' and v['issues']]
        missing = [x['case_id'] for x,v in rows if v['status']!='measured']
        regressions = []
        for x,v in rows:
            if x['case_id'] in pilot['groups']['normal_train'] and 'v1_dice' in x and v['status']=='measured':
                for m in ('T1w','T2w'):
                    delta = v['metrics'][m]['final']['dice'] - x['v1_dice'][m]
                    if delta < -.02:
                        regressions.append({'case_id':x['case_id'],'modality':m,'dice_delta':delta})
        results['candidate_summaries'][candidate] = {'measured_cases':len(complete),
            'missing_cases':missing,'cases_with_policy_issues':failures,
            'issue_counts':dict(Counter(issue for x in failures for issue in x['issues'])),
            'normal_control_regressions_over_0_02':regressions,
            'dice_min_median_mean': [float(min(dice)),float(np.median(dice)),float(np.mean(dice))] if dice else None,
            'original_nonnegative_case_count':sum(1 for x,v in rows if v['status']=='measured' and x['case_id'] not in pilot['negative_intensity_adapter_cases']),
            'original_nonnegative_dice_mean':float(np.mean([v['metrics'][m]['final']['dice']
                for x,v in rows if v['status']=='measured' and x['case_id'] not in pilot['negative_intensity_adapter_cases']
                for m in ('T1w','T2w')])) if any(v['status']=='measured' and x['case_id'] not in pilot['negative_intensity_adapter_cases'] for x,v in rows) else None,
            'negative_input_adapter_cases':pilot['negative_intensity_adapter_cases']}
    write_json(root/'pilot_summary.json', results)
    for candidate,s in results['candidate_summaries'].items():
        print(json.dumps({'candidate':candidate,'measured':s['measured_cases'],
            'missing':len(s['missing_cases']),'review_cases':len(s['cases_with_policy_issues']),
            'normal_regressions':len(s['normal_control_regressions_over_0_02']),
            'issue_counts':s['issue_counts'],'dice':s['dice_min_median_mean']}))


if __name__ == '__main__':
    main()
