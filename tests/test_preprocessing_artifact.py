import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.config import digest, file_hash
from src.preprocessing.artifact import FORMAT_VERSION, cache_path, validate_artifact_contract


def make_fixture(tmp_path):
    root = tmp_path
    image = root/'image.nii.gz'
    image.write_bytes(b'original image bytes')
    sample = SimpleNamespace(id='case-1', cohort='BraTS_GLI', image=str(image))
    cache_root = root/'cache'/'transform-v1'
    preprocessor = SimpleNamespace(version='transform-v1', cache=SimpleNamespace(root=cache_root))
    config = SimpleNamespace(
        project_root=str(root),
        preprocessing=OmegaConf.create({'spacing_policy':'native_grid','cache_dir':'cache', 'cache_enabled':True}),
        data=SimpleNamespace(target_version='targets-v1', datasets=['brats']),
    )
    image_sha = file_hash(image)
    key = digest({'image':image_sha,'cohort':sample.cohort,'preprocessing':preprocessor.version})
    tensor = cache_path(preprocessor,key)
    tensor.parent.mkdir(parents=True)
    tensor.write_bytes(b'cached patch tensor')

    manifest = root/'saved'/'preprocessing_manifest.jsonl'
    manifest.parent.mkdir()
    row = {'id':sample.id,'image_sha256':image_sha,'cache_key':key,'tokens':12}
    manifest.write_text(json.dumps(row)+'\n')
    qa_file=manifest.parent/'qa_summary.json'
    qa_file.write_text(json.dumps({'status':'complete','samples':1}))
    contract = {
        'preprocessing_version':preprocessor.version,
        'preprocessing_options':{'spacing_policy':'native_grid'},
        'target_version':config.data.target_version,
        'dataset_scope':['brats'],
        'input_manifest_hashes':{'train':'train','val':'val','test':'test'},
        'cache_root':'cache/transform-v1',
        'splits':{'train':{
            'samples':1,
            'manifest_path':'saved/preprocessing_manifest.jsonl',
            'manifest_sha256':file_hash(manifest),
            'qa_summary_sha256':file_hash(qa_file),
            'cache_entries':[{'id':sample.id,'cohort':sample.cohort,'image_sha256':image_sha,
                              'cache_key':key,'tokens':12}],
        }},
    }
    artifact = root/'artifact.json'
    artifact.write_text(json.dumps({'format_version':FORMAT_VERSION,'status':'complete',
                                    'artifact_id':digest(contract),'contract':contract}))
    summary = {name:{'sha256':name} for name in ('train','val','test')}
    return config, preprocessor, sample, summary, artifact


def test_reusable_artifact_matches_raw_image_and_cache(tmp_path):
    config, preprocessor, sample, summary, artifact = make_fixture(tmp_path)
    result = validate_artifact_contract(config,preprocessor,[sample],summary,'train',artifact)
    assert result['samples']==1 and result['cache_entries_present']==1


def test_reusable_artifact_rejects_changed_image(tmp_path):
    config, preprocessor, sample, summary, artifact = make_fixture(tmp_path)
    Path(sample.image).write_bytes(b'changed image bytes')
    with pytest.raises(ValueError,match='Input image or cache identity changed'):
        validate_artifact_contract(config,preprocessor,[sample],summary,'train',artifact)


def test_reusable_artifact_rejects_changed_settings(tmp_path):
    config, preprocessor, sample, summary, artifact = make_fixture(tmp_path)
    config.preprocessing.spacing_policy='2mm'
    with pytest.raises(ValueError,match='preprocessing options'):
        validate_artifact_contract(config,preprocessor,[sample],summary,'train',artifact)


def test_reusable_artifact_rejects_changed_dataset_scope(tmp_path):
    config, preprocessor, sample, summary, artifact = make_fixture(tmp_path)
    config.data.datasets=['isles']
    with pytest.raises(ValueError,match='dataset scope'):
        validate_artifact_contract(config,preprocessor,[sample],summary,'train',artifact)
