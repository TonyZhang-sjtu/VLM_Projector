import json
from pathlib import Path
from src.cli.common import parser,config_from
from src.config import path_of,file_hash
from src.data.manifest import load_manifests
from src.data.dataset import VolumeDataset
from src.preprocessing.registry import build_preprocessor
from src.models.multimodal import MultimodalModel
from src.training.checkpoint import provenance,load_checkpoint
from src.training.trainer import seed_all,validate
from src.evaluation.generate import predict
from src.evaluation.metrics import evaluate_rows


def main():
    cli=parser('Generate findings and report full-coverage, stratified metrics')
    cli.add_argument('--limit',type=int)
    cli.add_argument('--ids-file',help='JSON array of IDs belonging to the evaluation split')
    cli.add_argument('--control',choices=['zero','shuffle','text'])
    cli.add_argument('--skip-answer-ce',action='store_true',help='Skip the extra teacher-forced CE pass when only report metrics are needed')
    args=cli.parse_args();c=config_from(args,dataset_key='evaluation.datasets')
    if not c.evaluation.checkpoint or not c.evaluation.output_dir:cli.error('Set evaluation.checkpoint and evaluation.output_dir')
    output=path_of(c,c.evaluation.output_dir)
    if (output/'predictions.jsonl').exists():raise FileExistsError('Choose a new evaluation output directory')
    seed_all(c.seed)
    splits,summary=load_manifests(c,datasets=c.evaluation.datasets);samples=splits[c.evaluation.split]
    total=len(samples)
    if args.ids_file:
        ids=json.loads(path_of(c,args.ids_file).read_text());lookup={s.id:s for s in samples}
        if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or any(i not in lookup for i in ids):
            raise ValueError('Evaluation IDs must be unique, nonempty, and belong to the requested split')
        samples=[lookup[i] for i in ids]
    if args.limit:samples=samples[:args.limit]
    pre=build_preprocessor(c);data=VolumeDataset(samples,pre)
    from src.models.projectors.budget import prepare_budget
    budget_preflight=prepare_budget(c,samples)
    model=MultimodalModel(c).to(c.training.device)
    proof=provenance(c,model,pre,summary)
    checkpoint=path_of(c,c.evaluation.checkpoint)
    load_checkpoint(checkpoint,model,expected_provenance=proof)
    if budget_preflight:
        output.mkdir(parents=True,exist_ok=True)
        (output/'budget_preflight.json').write_text(json.dumps(budget_preflight,indent=2))
    rows=predict(model,data,output/'predictions.jsonl',file_hash(checkpoint),c.evaluation.batch_size,args.control)
    exclusions=[]
    if c.evaluation.review_exclusion_file:
        exclusions=json.loads(path_of(c,c.evaluation.review_exclusion_file).read_text())
        if not isinstance(exclusions,list) or not all(isinstance(x,str) for x in exclusions):
            raise ValueError('Review exclusion file must be a frozen JSON array of IDs')
    result=evaluate_rows(rows,[s.id for s in samples],c.evaluation.bootstrap_repeats,c.seed,exclusions)
    result.update(split_population=total,evaluated_subset=len(samples),
                  scope='subset_smoke' if len(samples)!=total else 'full_split',control=args.control,
                  training_dataset_scope=list(c.data.datasets),
                  evaluation_dataset_scope=summary[c.evaluation.split]['datasets'],
                  split_dataset_counts=summary[c.evaluation.split]['dataset_counts'])
    if args.control is None and not args.skip_answer_ce:result['answer_ce']=validate(model,data)
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result['coverage']['empty']:raise RuntimeError('Empty generations saved; evaluation is not a completed result')


if __name__=='__main__':main()
