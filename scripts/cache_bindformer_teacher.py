"""Export frozen pretrained NeuroVFM targets, never the trainable LoRA encoder.

python -m scripts.cache_bindformer_teacher --config PATH --output PATH --device cuda:0
Only selected train/val images are exported by default; no decoder is loaded.
"""
import argparse
import json
from pathlib import Path
import torch
from tqdm.auto import tqdm
from src.config import load_config, path_of
from src.data.manifest import load_manifests
from src.preprocessing.registry import build_preprocessor
from src.models.encoders.registry import build_encoder
from src.training.feature_targets import initialize_target_cache
from src.training.reproducibility import configure_reproducibility
from src.training.trainer import select_training_samples, seed_all


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config',action='append',required=True)
    cli.add_argument('--output',required=True)
    cli.add_argument('--device',default='cuda:0')
    cli.add_argument('--splits',nargs='+',choices=('train','val'),default=['train','val'])
    cli.add_argument('--limit',type=int,help='Acceptance subset only; omit to export the selected populations')
    args=cli.parse_args()
    if args.limit is not None and args.limit < 1:
        cli.error('--limit must be positive')
    config=load_config(args.config)
    # Preserve preprocessing/input protocol while building a clean frozen base.
    config.training.stage='a'
    config.training.feature_cache=False
    config.training.device=args.device
    seed_all(config.seed);configure_reproducibility(config)
    samples,_=load_manifests(config)
    preprocessor=build_preprocessor(config)
    encoder=build_encoder(config).to(args.device).eval()
    assert not any(p.requires_grad for p in encoder.parameters()) and not encoder.lora_targets
    store=initialize_target_cache(path_of(config,args.output),encoder.fingerprint,
                                  encoder.feature_dim,preprocessor.version)
    exported,cached=0,0
    for split in args.splits:
        selected=select_training_samples(config,samples[split],split)
        if args.limit is not None:selected=selected[:args.limit]
        for sample in tqdm(selected,desc=f'frozen teacher {split}'):
            volume=preprocessor(sample)
            path,_=store.target_path(volume.metadata)
            if path.exists():
                store.load(volume.coords,volume.metadata);cached+=1
                continue
            with torch.no_grad():
                encoded=encoder([volume])
            store.save(encoded.features[0],encoded.coords[0],volume.metadata)
            exported+=1
    print(json.dumps({'exported':exported,'already_cached':cached,
                      'contract':store.provenance()},indent=2))


if __name__=='__main__':main()
