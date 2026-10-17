"""Portable FT shell arguments; no GPU or pretrained assets required."""
import json
import os
import subprocess
import sys

import pytest

from src.config import ROOT


@pytest.mark.parametrize('index,lr,hold,epochs',[
    (1,'2e-5',0,20),(2,'2e-5',500,10),(3,'2e-5',500,20),
    (4,'3e-5',0,10),(5,'3e-5',500,10),(6,'5e-5',0,10)])
def test_entrypoint_arguments_from_another_directory(tmp_path,index,lr,hold,epochs):
    capture=tmp_path/'capture_python'
    capture.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                       'print(json.dumps({"args":sys.argv[1:],"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))\n')
    capture.chmod(0o700)
    local=tmp_path/'local_paths.yaml';local.write_text('{}\n')
    env=dict(os.environ,PROJECTOR_PYTHON=str(capture),PROJECTOR_LOCAL_CONFIG=str(local),
             CUDA_VISIBLE_DEVICES='7',FT_RUN_TAG='portable_test')
    script=ROOT/f'scripts/ft_ablation/e{index}.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    subprocess.run(['bash','-n',str(script.parent/'_common.bash')],check=True)
    result=subprocess.run(['bash',str(script),'--dry-run'],cwd=tmp_path,env=env,
                          text=True,capture_output=True,check=True)
    value=json.loads(result.stdout);args=value['args']
    assert value['gpu']=='7'
    assert args[:3]==['-u','-m','src.cli.baseline_sweep']
    assert args[-11:]==['--projector','avgpool3d','--ratios','8','--eval-split','test',
                       '--metrics','all','--eval-sample-fraction','1.0','--require-full-test']
    assert '--dry-run' in args
    assert args[args.index('--run-tag')+1]=='portable_test'
    configs=[args[i+1] for i,x in enumerate(args) if x=='--config']
    assert configs==[str(local),str(ROOT/'configs/ft_ablation/avgpool3d_r8.yaml')]
    settings=[args[i+1] for i,x in enumerate(args) if x=='--set']
    assert settings==[f'training.projector_lr={lr}',f'training.encoder_lr={lr}',
                      'training.projector_warmup_steps=77',f'training.projector_hold_steps={hold}',
                      'training.joint_warmup_steps=77',f'training.max_epochs={epochs}']


@pytest.mark.parametrize('explicit',[False,True])
def test_conda_python_precedes_path_and_explicit_override_wins(tmp_path,explicit):
    prefix=tmp_path/'projector'
    (prefix/'bin').mkdir(parents=True)
    for path,label in [(prefix/'bin/python','conda'),(tmp_path/'explicit_python','explicit')]:
        path.write_text('#!'+sys.executable+'\nimport json,sys\n'
                        f'print(json.dumps({{"selected":{label!r},"args":sys.argv[1:]}}))\n')
        path.chmod(0o700)
    env=dict(os.environ,CONDA_PREFIX=str(prefix),CUDA_VISIBLE_DEVICES='0')
    env.pop('PROJECTOR_PYTHON',None)
    env.pop('PROJECTOR_LOCAL_CONFIG',None)
    if explicit:env['PROJECTOR_PYTHON']=str(tmp_path/'explicit_python')
    result=subprocess.run(['bash',str(ROOT/'scripts/ft_ablation/e1.sh'),'--dry-run'],
                          cwd=tmp_path,env=env,text=True,capture_output=True,check=True)
    value=json.loads(result.stdout)
    assert value['selected']==('explicit' if explicit else 'conda')
    assert value['args'][:3]==['-u','-m','src.cli.baseline_sweep']
