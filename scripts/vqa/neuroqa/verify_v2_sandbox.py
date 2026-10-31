#!/usr/bin/env python3
"""Validate genuine v2 sandbox N4 outputs and exact patch-cache reloads."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from src.config import load_config
from src.data.neuroqa_manifest import write_json
from src.preprocessing.registry import build_preprocessor


def main():
    torch.set_num_threads(1)
    plan = ROOT/'results/vqa/neuroqa/preprocessing_v2_plan'
    root = plan/'sandbox/neuroqa_mni1mm_v2'
    config = load_config([ROOT/'configs/vqa/neuroqa/mixed_v2.yaml'])
    config.preprocessing.cache_dir = str(root)
    pre = build_preprocessor(config)
    rows = []
    for case in ('BraTS-GLI-00017-000','BraTS-GLI-00540-000'):
        folder = root/'cases'/case
        record = json.loads((folder/'complete.json').read_text())
        for m in ('T1w','T2w'):
            sample = SimpleNamespace(image=str(folder/f'{m}_mni_n4.nii.gz'),
                mask=str(folder/f'{m}_mni_mask.nii.gz'),cohort='BraTS_GLI')
            first = pre(sample)
            second = pre(sample)
            assert first.patches.shape == (9555,1024) and first.metadata['grid_dhw'] == [49,15,13]
            assert torch.equal(first.patches,second.patches) and torch.equal(first.coords,second.coords)
            assert torch.isfinite(first.patches).all() and not first.metadata['extra_resample_4mm']
            rows.append({'case_id':case,'modality':m,'tokens':9555,
                         'negative_voxels':record['qc'][m]['intensity_repair']['negative_voxels'],
                         'candidate':record['selected_candidate'], 'transform_kind':record['transform_kind']})
    assert next(x['negative_voxels'] for x in rows if x['case_id']=='BraTS-GLI-00540-000' and x['modality']=='T2w') == 26
    result = {'status':'passed', 'scope':'sandbox_only_not_production', 'volumes':rows,
              'preprocessing_version':pre.version,'same_process_cache_reload_exact':True}
    write_json(plan/'sandbox_verified.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
