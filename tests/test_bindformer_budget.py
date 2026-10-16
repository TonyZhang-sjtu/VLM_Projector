import math
from fractions import Fraction
import pytest
from src.config import load_config, save_config
from src.models.projectors.bindformer.budget import allocate_tokens
from src.models.projectors.budget import resolve_budget


@pytest.mark.parametrize('ratio,m,g,k', [(4,2194,550,411), (8,1097,273,206),
                                      (16,549,137,103), (32,275,67,52)])
def test_confirmed_budget_table_and_reload(ratio, m, g, k, tmp_path):
    c = load_config(['configs/projector/bindformer3d.yaml'],
                    [f'projector.target_compression_ratio={ratio}'])
    b = resolve_budget(c.projector, (39,15,15), (156,240,240))
    assert (b['expected_output_tokens'], b['global_tokens'], b['num_slots']) == (m,g,k)
    assert m == g+4*k == math.ceil(8775/ratio)
    assert c.projector.params.slot_iterations == 3
    assert c.projector.params.vocabulary_scope == 'brats_train'
    p = tmp_path/'config.yaml'
    save_config(c,p)
    assert dict(load_config([p]).projector.resolved_budget) == b
    with pytest.raises(ValueError, match='budget differs'):
        load_config([p], ['projector.params.global_fraction=0.125'])


def test_nearest_feasible_integer_oracle_and_tie():
    for m in range(5, 160):
        for alpha in (0, .125, .25, .5, .9):
            a = allocate_tokens(m,alpha)
            assert a['global_tokens']+4*a['num_slots'] == m
            possibilities = [(abs(Fraction(g,m)-Fraction(str(alpha))), -g, k)
                             for k in range(1, m//4+1)
                             if (g:=m-4*k) >= int(alpha>0)]
            assert a['num_slots'] == min(possibilities)[2]
    assert allocate_tokens(12,.5)['global_tokens'] == 8


@pytest.mark.parametrize('override', [
    'global_fraction=1', 'global_fraction=-0.1', 'global_fraction=true',
    'slot_iterations=0', 'support_beta=[1,2]', 'support_beta=-1',
    'geometry_space=atlas', 'within_slot_sa_depth=0', 'local_global_gate_init=1',
    'text_loss_weight=1', 'feature_loss_weight=-1', 'tuple_loss_weight=1',
    'position_mode=physical_fourier', 'roles=[finding]',
])
def test_invalid_or_unimplemented_modes_fail_early(override):
    with pytest.raises(ValueError):
        load_config(['configs/projector/bindformer3d.yaml'], [f'projector.params.{override}'])
