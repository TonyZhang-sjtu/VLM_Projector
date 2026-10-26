from copy import deepcopy
import pytest
from scripts.sync_asset_fingerprint import fingerprint
from scripts.check_sync_plans import normalized_config, normalized_command


def test_asset_identity_allows_relocation_but_detects_changed_bytes(tmp_path):
    one, two = tmp_path / 'server1', tmp_path / 'server2'
    one.mkdir(); two.mkdir()
    for root in (one, two):
        (root / 'weights').write_bytes(b'original')
    (two / '.cache').mkdir()
    (two / '.cache' / 'download').write_bytes(b'local bookkeeping')
    assert fingerprint(one) == fingerprint(two)
    (two / 'weights').write_bytes(b'modified')
    assert fingerprint(one) != fingerprint(two)
    (two / 'weights').unlink()
    with pytest.raises(ValueError, match='empty'):
        fingerprint(two)


def test_asset_symlinks_are_hashed_and_cycles_rejected(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(); target.mkdir()
    (source / 'model').write_bytes(b'weights')
    (target / 'linked').symlink_to(source, target_is_directory=True)
    assert fingerprint(target)['files'] == 1
    (source / 'cycle').symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match='cycle'):
        fingerprint(target)


def test_generated_runtime_paths_do_not_change_pristine_asset_identity(tmp_path):
    (tmp_path / 'model.tar.gz').write_bytes(b'pristine weights')
    expected = fingerprint(tmp_path)
    (tmp_path / 'runtime').mkdir()
    (tmp_path / 'runtime' / 'config.json').write_text('/server-specific/path')
    assert fingerprint(tmp_path, excluded_dirs={'runtime'}) == expected


def test_plan_normalization_keeps_numerical_and_entrypoint_changes():
    config = {'project_root': '/server1', 'seed': 42,
              'training': {'run_dir': '/server1/run', 'projector_lr': 0.0002,
                           'resume': None}}
    other = deepcopy(config)
    other['project_root'] = '/server2'
    other['training']['run_dir'] = '/server2/output'
    assert normalized_config(config) == normalized_config(other)
    other['training']['projector_lr'] = 0.0001
    assert normalized_config(config) != normalized_config(other)
    a = ['/python1', '-u', '/server1/experiments/report_output_v1/entry.py',
         'src.cli.train', '--config', '/server1/config.yaml']
    b = ['/python2', '-u', '/server2/experiments/report_output_v1/entry.py',
         'src.cli.train', '--config', '/server2/config.yaml']
    assert normalized_command(a) == normalized_command(b)
    b[2] = '/server2/experiments/report_facts_v1/entry.py'
    assert normalized_command(a) != normalized_command(b)
