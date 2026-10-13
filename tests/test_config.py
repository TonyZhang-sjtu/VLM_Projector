import pytest
from src.config import ROOT, load_config, accumulation_steps
from src.cli.common import config_from, parser
from src.preprocessing.registry import build_preprocessor


def test_composition_and_batch():
    c = load_config([ROOT / 'configs/training/stage_b_lora.yaml'])
    assert c.training.stage == 'b_lora' and not c.training.feature_cache
    assert accumulation_steps(c, 4) == 4
    with pytest.raises(ValueError):
        accumulation_steps(c, 3)


@pytest.mark.parametrize('override', ['training.typo=1', 'decoder.frozen=false',
    'training.stage=other', 'training.stage=b_lora', 'preprocessing.throughplane_mm=0',
    'training.microbatch_per_device=0', 'task.truncate_inputs=true', 'training.deterministic=maybe'])
def test_invalid_settings_fail(override):
    with pytest.raises(ValueError):
        load_config(overrides=[override])


def test_one_argument_spacing_switch():
    for policy, mm in [('native_grid',4),('2mm',2),('4mm',4)]:
        c=load_config(overrides=[f'preprocessing.spacing_policy={policy}'])
        assert c.preprocessing.throughplane_mm == mm
        assert c.preprocessing.boundary_policy == 'minimal_pad'


def test_stage_a_uses_debugged_learning_rate_and_absolute_warmup():
    c=load_config([ROOT/'configs/training/stage_a.yaml'])
    assert c.training.projector_lr==1e-5 and c.training.warmup_steps==4


def test_two_gpu_ddp_preset_keeps_global_batch():
    c=load_config([ROOT/'configs/training/stage_a.yaml',ROOT/'configs/training/deterministic.yaml',
                   ROOT/'configs/training/ddp.yaml'])
    assert c.training.distributed=='ddp'
    assert c.training.microbatch_per_device==1 and c.training.effective_batch_size==16
    assert accumulation_steps(c,world_size=2)==8


def test_shared_preprocessing_cache_cli_and_stable_version():
    cli = parser('test cache switch')
    enabled = config_from(cli.parse_args([]))
    explicit_enabled = config_from(cli.parse_args(['--use-cache']))
    disabled = config_from(cli.parse_args(['--no-use-cache']))
    alias_disabled = config_from(cli.parse_args(['--no-cache']))

    assert enabled.preprocessing.cache_enabled is True
    assert explicit_enabled.preprocessing.cache_enabled is True
    assert disabled.preprocessing.cache_enabled is False
    assert alias_disabled.preprocessing.cache_enabled is False
    assert enabled.preprocessing.cache_dir == 'assets/preprocessing'
    expected_version = '2c90586eacc0646f9bf4670fa85fce866db8a4874b88e4e47926ed2cfe1532b9'
    assert build_preprocessor(enabled).version == expected_version
    assert build_preprocessor(explicit_enabled).version == expected_version
    assert build_preprocessor(disabled).version == expected_version
