import json
import os
from omegaconf import OmegaConf
from src.cli.common import parser,config_from
from src.training.trainer import Trainer,training_budget,select_training_samples


def main():
    cli=parser('Train continuous joint LoRA, or legacy independent A/B experiments')
    cli.add_argument('--dry-run',action='store_true',help='Resolve selected data and update schedule without loading models')
    args=cli.parse_args()
    config=config_from(args)
    if config.task.name == 'neuroqa':
        if args.dry_run:
            raise ValueError('Use src.cli.vqa_train --dry-run for balanced QA planning')
        from src.training.vqa_trainer import VQATrainer
        VQATrainer(config).train()
        return
    if args.dry_run:
        from src.data.manifest import load_manifests
        from src.training.schedule import resolve_joint_schedule
        samples,_=load_manifests(config)
        train=select_training_samples(config,samples['train'],'train')
        windows,total=training_budget(config,len(train),int(os.getenv('WORLD_SIZE','1')))
        plan=resolve_joint_schedule(config.training,total).to_dict() if config.training.stage=='joint_lora' else None
        print(json.dumps({'config':OmegaConf.to_container(config,resolve=True),
            'train_samples':len(train),'updates_per_epoch':windows,'total_steps':total,'schedule':plan},indent=2))
        return
    trainer=Trainer(config)
    print(json.dumps(trainer.report,indent=2),flush=True)
    trainer.train()


if __name__=='__main__':main()
