"""Audit all generated control records against immutable Normal predictions."""
from collections import Counter,defaultdict
import math

from .common import OUT,entries,read,rows,write
from src.evaluation.clinical import file_digest


def main():
    _,es=entries();root=OUT/'visual';plan=read(root/'plan.json');checked=[]
    for setting in plan['settings']:
        folder=root/setting.replace('/','__');done=read(folder/'complete.json')
        original={r['id']:r for r in rows(es[setting]['paths']['predictions'])}
        normal={r['id']:r for r in rows(folder/'normal/predictions.jsonl')}
        assert len(original)==len(normal)==560 and original.keys()==normal.keys()
        for key in original:
            for field in ('prediction','token_ids','reference','patient_id','sequence'):
                assert original[key][field]==normal[key][field],(setting,key,field)
        bag=Counter((r['sequence'],r['prediction'],tuple(r['token_ids'])) for r in normal.values())
        conditions={}
        for name,sha in done['outputs'].items():
            path=folder/name;assert file_digest(path)==sha
            rr=rows(path);condition=path.parent.name
            assert len(rr)==len({r['id'] for r in rr})==560 and {r['id'] for r in rr}==set(normal)
            for r in rr:
                target=normal[r['id']]
                for field in ('reference','patient_id','sequence','answer_tokens'):
                    assert r[field]==target[field],(setting,condition,r['id'],field)
                assert math.isfinite(r['loss_sum']) and r['loss_sum']>=0 and r['answer_tokens']>0
                if condition=='zero':assert r['visual_token_count']==target['visual_token_count']
                if condition=='text':assert r['visual_token_count']==0
            if condition.startswith('shuffle_'):
                assert Counter((r['sequence'],r['prediction'],tuple(r['token_ids'])) for r in rr)==bag
                patient_donors=defaultdict(set);seed=condition.split('_')[1]
                for r in rr:
                    donor=original[plan['donors'][seed][r['id']]]
                    assert r['image_donor_id']==donor['id']
                    assert r['donor_patient_id']==donor['patient_id']!=r['patient_id']
                    assert r['sequence']==donor['sequence'] and r['cohort']==donor['cohort']
                    assert r['prediction']==donor['prediction'] and r['token_ids']==donor['token_ids']
                    patient_donors[r['patient_id']].add(donor['patient_id'])
                assert all(len(v)==1 for v in patient_donors.values())
            conditions[condition]=dict(rows=len(rr),answer_tokens=sum(r['answer_tokens'] for r in rr),sha256=sha)
        assert len(conditions)==6
        checked.append(dict(setting=setting,conditions=conditions))
    write(root/'inference_validation.json',dict(status='passed',settings=len(checked),rows=8*6*560,
          plan_sha256=file_digest(root/'plan.json'),checks=checked,
          invariants=['original Normal texts unchanged','target references and answer-token counts fixed',
                      'same-sequence donor output multiset preserved','patient-consistent derangements',
                      'zero/text actual prefix lengths','all CE finite and nonnegative']))
    print('VISUAL_INFERENCE_AUDIT_PASSED',len(checked),'settings; 26880 records',flush=True)


if __name__=='__main__':main()
