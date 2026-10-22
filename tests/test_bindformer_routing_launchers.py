"""Compare the actual Exp1020 command/config with the user's 1019 reference."""
import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import pytest
from src.config import ROOT
from src.cli import baseline_sweep as sweep
from test_baseline_sweep import population

ROUTES = ['global_routing', 'fixed_region_mean', 'fixed_region_adaptive']
INTERACTIONS = {'saca': 'sa_ca', 'va': 'vanilla'}


def capture(tmp_path, script, extra=(), changes=None, return_record=False):
    stub=tmp_path/'python stub'
    stub.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                    'print(json.dumps({"args":sys.argv[1:],'
                    '"gpu":os.environ.get("CUDA_VISIBLE_DEVICES")}))\n')
    stub.chmod(0o755)
    local=tmp_path/'local.yaml';local.write_text('{}\n')
    env={k:v for k,v in os.environ.items()
         if not k.startswith(('BIND_', 'PROJECTOR_')) and k!='WORLD_SIZE'}
    env.update(PROJECTOR_PYTHON=str(stub),PROJECTOR_LOCAL_CONFIG=str(local),CUDA_VISIBLE_DEVICES='3')
    env.update(changes or {})
    result=subprocess.run(['bash',str(script),'--dry-run',*extra],cwd=tmp_path,
                          env=env,check=True,capture_output=True,text=True)
    record=json.loads(result.stdout)
    args=sweep.parser().parse_args(record['args'][3:])
    return (args,record) if return_record else args


@pytest.mark.parametrize('variant',ROUTES)
@pytest.mark.parametrize('interaction',INTERACTIONS)
def test_full_recipe_matches_reference_except_switches_and_run_name(
        tmp_path,population,variant,interaction):
    ref=ROOT/f'scripts/baselines/BindFormer/run_bindformer_{INTERACTIONS[interaction]}_constant_lr_1019.sh'
    new=ROOT/f'scripts/ft_ablation/Exp1020/run_bindformer_{interaction}_{variant}_constlr.sh'
    # The user edits each entry's default array independently; compare the same
    # explicit budgets instead of requiring every file to have identical defaults.
    ratios=['--ratios','8','16','32','64']
    baseline=sweep.build_plan(capture(tmp_path,ref,ratios))
    plan=sweep.build_plan(capture(tmp_path,new,ratios))
    sweep.validate_full_test_plan(plan)
    assert plan['ratios']==baseline['ratios']==[8,16,32,64]
    assert f'{interaction}_{variant}_constlr_' in plan['run_root']
    for left,right in zip(baseline['runs'],plan['runs']):
        assert left['schedule']==right['schedule']
        assert left['data_identity']==right['data_identity']
        assert left['budget']==right['budget']
        ca,cb=copy.deepcopy(left['config']),copy.deepcopy(right['config'])
        for c in (ca,cb):c['training'].pop('run_dir')
        p=cb['projector']['params']
        assert p['within_slot_sa']==p['local_reads_global']==(interaction=='saca')
        assert p['router_value_normalization']==(variant!='global_routing')
        assert p['fixed_spatial_regions']==(variant!='global_routing')
        assert p['spatial_mean_aggregation']==(variant=='fixed_region_mean')
        assert p['spatial_adaptive_aggregation']==(variant=='fixed_region_adaptive')
        for key in ('router_value_normalization','fixed_spatial_regions',
                    'spatial_mean_aggregation','spatial_adaptive_aggregation'):
            ca['projector']['params'].pop(key);cb['projector']['params'].pop(key)
        assert ca==cb
        assert right['evaluation_samples']==560 and Path(right['checkpoint']).name=='best.pt'
    assert not Path(plan['run_root']).exists()


def test_switches_remain_independent_and_cli_overrides_win(tmp_path,population):
    new=ROOT/'scripts/ft_ablation/Exp1020/run_bindformer_saca_fixed_region_mean_constlr.sh'
    args=capture(tmp_path,new,[
        '--ratios','8','--set','projector.params.spatial_mean_aggregation=false',
        '--set','projector.params.spatial_adaptive_aggregation=true'],
        {'BIND_ROUTER_VALUE_NORMALIZATION':'false'})
    plan=sweep.build_plan(args)
    p=plan['runs'][0]['config']['projector']['params']
    assert not p['router_value_normalization'] and p['fixed_spatial_regions']
    assert not p['spatial_mean_aggregation'] and p['spatial_adaptive_aggregation']


@pytest.mark.parametrize('variant',ROUTES)
@pytest.mark.parametrize('interaction',INTERACTIONS)
def test_entrypoint_edits_reach_executor_without_common_reset(
        tmp_path,population,variant,interaction):
    directory=ROOT/'scripts/ft_ablation/Exp1020'
    source=directory/f'run_bindformer_{interaction}_{variant}_constlr.sh'
    edited=tmp_path/source.name
    contents=source.read_text()
    changes={
        re.search(r'\$\{CUDA_VISIBLE_DEVICES:-\d*\}',contents).group():'${CUDA_VISIBLE_DEVICES:-2}',
        re.search(r'^projector_ratios=\([^\n)]*\)',contents,re.M).group():'projector_ratios=(16)',
        'projector_lr=2e-4':'projector_lr=3e-4',
        'encoder_lr=2e-4':'encoder_lr=1e-4',
        'projector_max_epochs=20':'projector_max_epochs=5',
        'projector_effective_batch_size=16':'projector_effective_batch_size=8',
        'projector_microbatch_per_device=1':'projector_microbatch_per_device=2',
        'encoder_lora_rank=16':'encoder_lora_rank=8',
        'encoder_lora_alpha=32':'encoder_lora_alpha=16',
        f'bind_within_slot_sa={str(interaction=="saca").lower()}':'bind_within_slot_sa=false',
        f'bind_local_reads_global={str(interaction=="saca").lower()}':'bind_local_reads_global=true',
        'bind_head_type=mlp':'bind_head_type=linear',
        '${BIND_ROUTER_VALUE_PROJECTION:-false}':'${BIND_ROUTER_VALUE_PROJECTION:-true}',
        '${BIND_SPATIAL_TEMPERATURE:-1.0}':'${BIND_SPATIAL_TEMPERATURE:-0.5}',
        'source "$routing_script_dir/_common_constlr.bash"':
            f'source {shlex.quote(str(directory/"_common_constlr.bash"))}',
    }
    for before,after in changes.items():
        assert contents.count(before)==1
        contents=contents.replace(before,after)
    edited.write_text(contents)
    # Capture Python is a stub: checking GPU selection never opens a GPU.
    args,record=capture(tmp_path,edited,changes={'CUDA_VISIBLE_DEVICES':''},
                        return_record=True)
    assert record['gpu']=='2'
    plan=sweep.build_plan(args)
    sweep.validate_full_test_plan(plan)
    assert plan['ratios']==[16]
    c=plan['runs'][0]['config'];t=c['training'];p=c['projector']['params']
    assert t['projector_lr']==3e-4 and t['encoder_lr']==1e-4
    assert t['max_epochs']==5
    assert t['effective_batch_size']==8 and t['microbatch_per_device']==2
    assert t['lora']['rank']==8 and t['lora']['alpha']==16
    assert not p['within_slot_sa'] and p['local_reads_global']
    assert p['head_type']=='linear' and p['router_value_projection']
    assert p['spatial_attention_temperature']==0.5
    assert p['fixed_spatial_regions']==(variant!='global_routing')
    assert not Path(plan['run_root']).exists()

    # External GPU and explicit CLI options still take precedence over file edits.
    args,record=capture(tmp_path,edited,[
        '--ratios','32','--run-tag','explicit_override',
        '--set','training.projector_lr=4e-4',
        '--set','projector.params.within_slot_sa=true',
        '--set','projector.params.local_reads_global=false'],
        changes={'CUDA_VISIBLE_DEVICES':'3'},return_record=True)
    plan=sweep.build_plan(args)
    assert record['gpu']=='3' and plan['ratios']==[32]
    assert plan['runs'][0]['config']['training']['projector_lr']==4e-4
    assert plan['runs'][0]['config']['projector']['params']['within_slot_sa']
    assert not plan['runs'][0]['config']['projector']['params']['local_reads_global']
    assert args.run_tag=='explicit_override'
