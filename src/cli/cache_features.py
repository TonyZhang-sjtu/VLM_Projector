from pathlib import Path
import json
from src.cli.common import parser, config_from
from src.config import path_of
from src.data.manifest import load_manifests
from src.preprocessing.registry import build_preprocessor
from src.models.encoders.registry import build_encoder
from tqdm.auto import tqdm


def main():
    cli = parser('Cache fixed encoder features for Stage A only')
    cli.add_argument('--split', choices=['train','val','test'],default='train')
    cli.add_argument('--limit',type=int)
    args = cli.parse_args()
    c = config_from(args)
    if c.training.stage != 'a' or not c.training.feature_cache:
        cli.error('Feature caching requires Stage A with feature_cache=true')
    samples = load_manifests(c)[0][args.split]
    if args.limit:
        samples = samples[:args.limit]
    pre = build_preprocessor(c)
    encoder = build_encoder(c).to(c.training.device).eval()
    for sample in tqdm(samples,desc=f'cache {args.split}',unit='volume',dynamic_ncols=True):
        output = encoder([pre(sample)])


if __name__ == '__main__':
    main()
