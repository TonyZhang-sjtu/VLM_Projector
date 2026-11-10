"""Prepare a frozen local audit and RadMatch inputs, without running any judge.

Run from the project root with `python -m experiments.brain_location_audit_v1.prepare`.
Writes only a NEW output directory; never changes original predictions/scores.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

from .protocol import VERSION, extract_graph, label_sets, score_sets

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read_rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def serial_labels(extraction):
    labels = label_sets(extraction)
    return {k: sorted(v) if isinstance(v, set) else v for k, v in labels.items()}


def green_location_count(details):
    categories=details.get('error_categories',[])
    counts=details.get('significant_errors',[])
    if not isinstance(categories,list) or not isinstance(counts,list) or len(categories)!=len(counts):
        return None
    found=[value for name,value in zip(categories,counts)
           if 'anatomic location/position' in name.lower()]
    if len(found)!=1 or isinstance(found[0],bool) or not isinstance(found[0],(int,float)) or found[0]<0:
        return None
    return found[0]


def choose_pilot(references, patients=24):
    """Patient selection depends on IDs only, never method outputs or scores."""
    ids = sorted({r['patient_id'] for r in references.values()},
                 key=lambda p: digest('20261108/location-pilot/' + p))[:patients]
    sequences = ['T1W', 'T1C', 'T2W', 'FLAIR']
    chosen=[]
    for i, patient in enumerate(ids):
        rows=sorted((r for r in references.values() if r['patient_id']==patient), key=lambda r:r['id'])
        preferred=[r for r in rows if r['sequence']==sequences[i % 4]]
        chosen.append((preferred or rows)[0]['id'])
    return chosen


def stress_cases():
    """Author-written clinical-language probes, not a human-validated benchmark."""
    base='A lesion is present in the left frontal lobe.'
    pairs=[
        ('identity',base,base,'same location'),
        ('word_order',base,'The lesion is in the frontal lobe on the left.','same location'),
        ('paraphrase',base,'A mass is identified within the left frontal region.','same location'),
        ('missing_side',base,'A lesion is present in the frontal lobe.','site correct; side/joint incomplete'),
        ('side_flip',base,'A lesion is present in the right frontal lobe.','side/joint wrong; site correct'),
        ('site_flip',base,'A lesion is present in the left temporal lobe.','site/joint wrong; side correct'),
        ('negation',base,'No lesion is present in the left frontal lobe.','positive location must not match'),
        ('uncertain_presence',base,'A possible lesion is present in the left frontal lobe.','do not equate uncertain with definite presence'),
        ('uncertain_diagnosis',base,'A lesion in the left frontal lobe may represent a glioma.','location remains definite'),
        ('irrelevant_negation',base,'A left frontal lesion is present, without midline shift.','location remains definite'),
        ('ventricle_both',base,'A left frontal lesion compresses both lateral ventricles.','lesion remains left, not bilateral'),
        ('ventricle_side',base,'A left frontal lesion displaces the right lateral ventricle.','lesion remains left'),
        ('bilateral_overcall',base,'Lesions are present in both frontal lobes.','bilateral is not a correct left-only match'),
        ('bilateral_paraphrase','Bilateral frontal lesions are present.','Lesions are present in the left and right frontal lobes.','equivalent bilateral involvement'),
        ('multisite_order','There are lesions in the left frontal and right temporal lobes.','There are lesions in the right temporal and left frontal lobes.','same location set'),
        ('multisite_binding','There are lesions in the left frontal and right temporal lobes.','There are lesions in the right frontal and left temporal lobes.','side/site sets agree but joint pairs are wrong'),
        ('multisite_omission','There are lesions in the left frontal and right temporal lobes.',base,'missing right temporal location'),
        ('extra_location',base,'There are lesions in the left frontal and right temporal lobes.','extra right temporal location'),
        ('compound','A left frontoparietal lesion is present.','A lesion involves the left frontal and parietal lobes.','same location set'),
        ('brainstem','A lesion is present in the brainstem.','A lesion is present in the brain stem.','site matches; do not invent side'),
        ('insula','A lesion is present in the left insula.','A left insular lesion is present.','same location'),
        ('midline_lesion','A midline corpus callosum lesion is present.','A lesion in the corpus callosum is located at the midline.','same explicit midline lesion location'),
        ('midline_shift',base+' There is rightward midline shift.',base+' There is leftward midline shift.','lesion location unchanged; displacement direction is wrong'),
        ('midline_vs_bilateral','A midline lesion is present.','A bilateral lesion is present.','midline and bilateral differ'),
        ('unreported_reference','The lesion shows heterogeneous enhancement.',base,'reference does not establish a location; not negative truth'),
        ('duplicate',base,base+' The lesion is in the left frontal lobe.','duplicate mention is not an extra location'),
        ('lesion_vs_edema',base,'Edema is present in the left frontal lobe.','edema must not silently substitute for the target lesion'),
        ('parent_child','A lesion is present in the pons.','A lesion is present in the brainstem.','coarse compatible; fine anatomy less specific'),
    ]
    return [dict(id=f'stress_{i:02d}_{name}',reference=r,prediction=p,expected_property=e,
                 validation_status='author_written_probe_not_clinician_validated')
            for i,(name,r,p,e) in enumerate(pairs)]


def add_sheet(wb, name, rows):
    ws=wb.create_sheet(name)
    if not rows:
        return
    keys=list(rows[0]);ws.append(keys)
    for row in rows:
        ws.append([json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],(list,dict)) else row[k] for k in keys])
    ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
    for cell in ws[1]:
        cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='17365D')
    for col in ws.columns:
        title=str(col[0].value)
        ws.column_dimensions[col[0].column_letter].width=80 if any(x in title for x in ('text','reference','prediction','facts','evidence')) else 28
        for cell in col:cell.alignment=Alignment(vertical='top',wrap_text=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=ROOT/'results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/brain_location_audit_1108_v1')
    args=parser.parse_args()
    idx=json.loads(args.index.read_text());entries=[e for e in idx['paper_settings'] if e['key'].startswith('rg/')]
    assert len(entries)==57, 'Expected 49 RG main + 8 RG ablations'
    master=Path(idx['master']);assert hashlib.sha256(master.read_bytes()).hexdigest()==idx['master_sha256']
    canonical=next(e for e in entries if e['key']=='rg/A0/r32')
    def clinical_path(e):return Path(e['paths']['metric']).parent/'scoring/clinical_scores.jsonl'
    refs={}
    for row in read_rows(clinical_path(canonical)):
        rg=row['clinical']['radgraph'];graph=rg.get('details',{}).get('reference_graph') if rg['status']=='ok' else None
        parsed=extract_graph(graph)
        refs[row['id']]=dict(id=row['id'],patient_id=row['patient_id'],sequence=row['sequence'],reference=row['reference'],
                             reference_sha256=digest(row['reference']),graph_sha256=digest(json.dumps(graph,sort_keys=True)),
                             extraction=parsed,text_audit_status=row.get('text_audit_status',''))
    assert len(refs)==560 and len({r['patient_id'] for r in refs.values()})==140
    args.output.mkdir(parents=True,exist_ok=False)
    review=[]
    for r in refs.values():
        flags=list(r['extraction']['flags'])
        if 'requires_review' in r['text_audit_status']:flags.append('source_text_requires_review')
        review.append(dict(id=r['id'],patient_id=r['patient_id'],sequence=r['sequence'],reference_text=r['reference'],
            reference_sha256=r['reference_sha256'],auto_facts=r['extraction']['facts'],
            auto_midline_shift=r['extraction']['midline_shift'],auto_labels=serial_labels(r['extraction']),
            review_flags=flags,source_audit_status=r['text_audit_status'],review_status='pending',reviewer='',
            gold_facts_json='',gold_side_eligible='',gold_site_eligible='',gold_joint_eligible='',
            gold_midline_shift_json='',evidence_and_notes=''))
    (args.output/'reference_candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in refs.values()))
    pilot=choose_pilot(refs);pilot_set=set(pilot)
    selected=[e for e in entries if e['key']=='rg/MLP/r1' or
              (e['key'].split('/')[1] in ['A0','AvgPool3D','Perceiver_NV','A1','A2','A3','A4'] and
               e['key'].split('/')[-1] in ['r32','r256'])]
    assert len(selected)==15
    selected_keys={e['key'] for e in selected}
    pack=args.output/'radmatch_pilot';gt=pack/'reports_gt';gt.mkdir(parents=True)
    public_id={rid:f'case_{i:03d}' for i,rid in enumerate(pilot)}
    for rid,name in public_id.items():(gt/f'{name}.txt').write_text(refs[rid]['reference'])
    blind_review=[];blind_map=[];summaries=[];provenance=[];green_examples=[]
    detailed=args.output/'draft_graph_diagnostics';detailed.mkdir()
    for e in entries:
        key=e['key'];tag=key.replace('/','_');count=0;stats={k:Counter() for k in ['side','site','joint']};green=Counter();seen=set()
        prediction_hashes={};graph_mismatches=0
        if key in selected_keys:(pack/'reports_pred'/tag).mkdir(parents=True)
        with (detailed/f'{tag}.jsonl').open('w') as fout:
            for row in read_rows(clinical_path(e)):
                rid=row['id'];assert rid not in seen;seen.add(rid);count+=1
                assert rid in refs and digest(row['reference'])==refs[rid]['reference_sha256']
                assert row['patient_id']==refs[rid]['patient_id']
                rg=row['clinical']['radgraph'];gd=rg.get('details',{})
                pred=extract_graph(gd.get('prediction_graph') if rg['status']=='ok' else None)
                # One frozen canonical reference label set for EVERY method.
                graph_mismatches+=int(digest(json.dumps(gd.get('reference_graph'),sort_keys=True))!=refs[rid]['graph_sha256'])
                scores=score_sets(refs[rid]['extraction'],pred)
                if 'requires_review' in refs[rid]['text_audit_status']:
                    scores={f:{'eligible':False,'reason':'source_text_requires_review'} for f in scores}
                for field,sc in scores.items():
                    if sc['eligible']:
                        stats[field].update({k:sc[k] for k in ['exact','tp','fp','fn','missing']});stats[field]['n']+=1
                fout.write(json.dumps(dict(id=rid,prediction_sha256=digest(row['prediction']),
                    prediction_graph=pred,scores=scores,status='unvalidated_draft_graph_agreement'),ensure_ascii=False)+'\n')
                g=row['clinical']['green']
                if g['status']=='ok':
                    categories=g['details'].get('error_categories',{})
                    # Preserve actual category schema; no ad hoc renaming into accuracy.
                    c=green_location_count(g['details'])
                    if isinstance(c,(int,float)):
                        green['valid_location_category']+=1;green['location_error_count']+=c
                        green['reports_with_location_error']+=int(c>0)
                    if c and len(green_examples)<12:
                        green_examples.append(dict(setting=key,id=rid,reference=row['reference'],prediction=row['prediction'],
                                                   categories=categories,analysis=g['details'].get('analysis')))
                else:green['scoring_failed']+=1
                if key in selected_keys and rid in pilot_set:
                    (pack/'reports_pred'/tag/f'{public_id[rid]}.txt').write_text(row['prediction'])
                    blind=digest('blind/'+key+'/'+rid)[:16]
                    blind_map.append(dict(blind_id=blind,setting=key,id=rid,patient_id=row['patient_id']))
                    blind_review.append(dict(blind_id=blind,reference_text=row['reference'],prediction_text=row['prediction'],
                        review_status='pending',reviewer='',reference_location_facts='',prediction_location_facts='',
                        side_error='',site_error='',joint_binding_error='',midline_shift_error='',notes=''))
                prediction_hashes[rid]=digest(row['prediction'])
        assert seen==set(refs) and count==560
        original_rows=list(read_rows(e['paths']['predictions']))
        assert len(original_rows)==560 and {r['id'] for r in original_rows}==seen
        assert {r['id']:digest(r['prediction']) for r in original_rows}==prediction_hashes
        assert all(digest(r['reference'])==refs[r['id']]['reference_sha256'] for r in original_rows)
        summary={'setting':key,'samples':count,'status':'unvalidated_draft_graph_agreement',
                 'reference_graph_disagreements':graph_mismatches,**dict(green)}
        for field,c in stats.items():
            summary.update({f'{field}_{k}':c[k] for k in ['n','exact','tp','fp','fn','missing']})
            summary[f'{field}_exact_rate']=c['exact']/c['n'] if c['n'] else None
            den=2*c['tp']+c['fp']+c['fn'];summary[f'{field}_micro_f1']=2*c['tp']/den if den else None
        summaries.append(summary)
        provenance.append(dict(setting=key,clinical_path=str(clinical_path(e)),
            metric_path=e['paths']['metric'],prediction_path=e['paths']['predictions'],
            prediction_texts_sha256=digest(json.dumps(prediction_hashes,sort_keys=True))))
        print('AUDITED',key,count,flush=True)
    for field in ['side_n','site_n','joint_n']:
        assert len({r[field] for r in summaries})==1, 'Method-dependent denominator'
    cases=stress_cases();stress=pack/'stress'
    for folder in ['reports_gt','reports_pred']:(stress/folder).mkdir(parents=True)
    for case in cases:
        (stress/'reports_gt'/f'{case["id"]}.txt').write_text(case['reference'])
        (stress/'reports_pred'/f'{case["id"]}.txt').write_text(case['prediction'])
    dump(args.output/'stress_cases.json',cases)
    dump(args.output/'pilot_identity_map.json',blind_map)
    dump(args.output/'source_provenance.json',provenance)
    dump(args.output/'green_location_examples.json',green_examples)
    wb=openpyxl.Workbook();wb.remove(wb.active)
    add_sheet(wb,'Instructions',[{'item':'Status','value':'All labels are automatic drafts; no human review has been performed.'},
        {'item':'Scope','value':'Lesion-location facts, not physical lesion instances or image-grounded truth.'},
        {'item':'Reference review','value':'Review all 560 texts; record positive/negative/uncertain facts and evidence. Do not infer unreported locations from predictions.'},
        {'item':'Prediction review','value':'360 pairs, 24 fixed patients, 15 settings; method identities are held separately.'},
        {'item':'Unknown','value':'Unknown reference fields are not negative labels; empty or negated predictions remain incorrect on eligible positive references.'},
        {'item':'Prior exposure','value':'Earlier analysis inspected some test reports; this is retrospective protocol development, not an untouched held-out metric validation.'},
        {'item':'Independent validation','value':'Validate on blinded clinician annotations and an independent set before a strong metric novelty/validity claim.'}])
    add_sheet(wb,'Reference_review',review)
    add_sheet(wb,'Prediction_review',sorted(blind_review,key=lambda r:r['blind_id']))
    add_sheet(wb,'Stress_tests',cases)
    wb.save(args.output/'location_review_560.xlsx')
    fields=list(dict.fromkeys(k for r in summaries for k in r))
    with (args.output/'draft_diagnostics_by_setting.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(summaries)
    manifest=dict(version=VERSION,reference_reports=560,patients=140,rg_settings=len(entries),
        evaluated_report_pairs=sum(r['samples'] for r in summaries),pilot_patients=24,pilot_settings=len(selected),
        pilot_pairs=len(blind_review),stress_pairs=len(cases),human_reviewed_reference_reports=0,
        radmatch_executed=False,clinical_models_loaded=False,source_workbook_sha256=idx['master_sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(Path(__file__).with_name('protocol.py').read_bytes()).hexdigest(),
        pilot_ids=pilot,pilot_settings_keys=sorted(selected_keys),
        reference_draft_eligible_counts={k:summaries[0][k] for k in ['side_n','site_n','joint_n']},
        limitations=['Automatic graph proposals are not gold annotations.',
                     'F1/exact match are established statistics, not a novel metric formula.',
                     'No lesion instance correspondence or imaging localization is established.',
                     'Do not infer unavailable reference locations.',
                     'RadMatch itself still requires an isolated Python >=3.11 environment and a judge endpoint.'])
    dump(args.output/'manifest.json',manifest)
    assert hashlib.sha256(master.read_bytes()).hexdigest()==idx['master_sha256']
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
