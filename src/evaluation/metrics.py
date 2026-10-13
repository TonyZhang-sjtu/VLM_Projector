import importlib.metadata as metadata
import numpy as np
from sacrebleu.metrics import BLEU
from rouge_score.rouge_scorer import RougeScorer
from tqdm.auto import tqdm


def normalize(text):return ' '.join(text.split())


def score(rows):
    if not rows:raise ValueError('Cannot score empty predictions')
    predictions=[normalize(r['prediction']) for r in rows]
    references=[normalize(r['reference']) for r in rows]
    bleu=BLEU(tokenize='13a',lowercase=False,smooth_method='exp',effective_order=False)
    value=bleu.corpus_score(predictions,[references]).score
    rouge=RougeScorer(['rougeL'],use_stemmer=True)
    per_sample=[100*rouge.score(ref,pred)['rougeL'].fmeasure for ref,pred in zip(references,predictions)]
    return {'samples':len(rows),'bleu4':value,'rougeL':float(np.mean(per_sample)),
            'bleu_signature':str(bleu.get_signature()),'rouge_per_sample':per_sample}


def patient_bootstrap(rows,repeats=1000,seed=42,paired_rows=None):
    if repeats<1:return {'status':'not_run','repeats':0}
    patients=sorted({r['patient_id'] for r in rows})
    if not patients:raise ValueError('No patients to bootstrap')
    grouped={p:[r for r in rows if r['patient_id']==p] for p in patients}
    pair=None
    if paired_rows is not None:
        pair={r['id']:r for r in paired_rows}
        if len(paired_rows)!=len(rows) or len(pair)!=len(rows) or set(pair)!={r['id'] for r in rows}:raise ValueError('Paired prediction ID mismatch')
        for row in rows:
            if any(pair[row['id']][k]!=row[k] for k in ('reference','patient_id')):
                raise ValueError('Paired reference/patient mismatch')
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in tqdm(range(repeats),desc='patient bootstrap',unit='resample',leave=False,dynamic_ncols=True,mininterval=2):
        sample=[r for patient in rng.choice(patients,size=len(patients),replace=True) for r in grouped[patient]]
        first=score(sample)
        vals=[first['bleu4'],first['rougeL']]
        if pair is not None:
            other=score([pair[r['id']] for r in sample]);vals=[other['bleu4']-vals[0],other['rougeL']-vals[1]]
        draws.append(vals)
    limits=np.percentile(np.array(draws),[2.5,97.5],axis=0)
    return {'unit':'patient_id','repeats':repeats,'seed':seed,'patients':len(patients),
            'quantity':'paired B minus A' if pair is not None else 'score',
            'bleu4_95ci':limits[:,0].tolist(),'rougeL_95ci':limits[:,1].tolist()}


def evaluate_rows(rows,expected_ids,repeats=1000,seed=42,excluded_ids=()):
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):raise ValueError('Prediction coverage mismatch or duplicate IDs')
    overall=score(rows)
    per_sample=overall.pop('rouge_per_sample')
    strata={}
    for field in ('cohort','sequence','source'):
        def key(row):return ('BraTS' if row['cohort'].startswith('BraTS_') else row['cohort']) if field=='source' else row[field]
        groups={name:[r for r in rows if key(r)==name] for name in sorted({key(r) for r in rows})}
        strata[field]={name:{k:v for k,v in score(group).items() if k!='rouge_per_sample'} for name,group in groups.items()}
    hierarchy={}
    for field in ('case_id','patient_id'):
        groups={}
        for row,value in zip(rows,per_sample):groups.setdefault(row[field],[]).append(value)
        hierarchy[field]={'units':len(groups),'macro_rougeL':float(np.mean([np.mean(v) for v in groups.values()]))}
    result={'overall':overall,'strata':strata,'hierarchy':hierarchy,
        'coverage':{'expected':len(expected_ids),'present':len(rows),
                    'empty':sum(not r['prediction'].strip() for r in rows),
                    'generation_limit':sum(r.get('hit_generation_limit',False) for r in rows)},
        'versions':{p:metadata.version(p) for p in ('sacrebleu','rouge-score','nltk')},
        'bootstrap':patient_bootstrap(rows,repeats,seed),
        'semantic_metrics':{'BERTScore':'not_run','RaTEScore':'not_run'}}
    result['status']='complete' if result['coverage']['empty']==0 else 'generation_failures'
    excluded=set(excluded_ids)
    if excluded:
        subset=[r for r in rows if r['id'] not in excluded]
        result['review_sensitivity']={'excluded_count':len(rows)-len(subset),
            'remaining_count':len(subset),'metrics':{k:v for k,v in score(subset).items() if k!='rouge_per_sample'} if subset else None}
    return result
