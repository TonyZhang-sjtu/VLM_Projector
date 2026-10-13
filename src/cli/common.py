import argparse
from src.config import load_config, validate
from src.data.datasets import normalize_datasets


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", action="append", default=[])
    result.add_argument("--set", action="append", default=[], dest="overrides")
    result.add_argument("--datasets", nargs="+", metavar="DATASET",
                        help="Dataset scope: brats, isles, wmh, all, or a combination")
    cache = result.add_mutually_exclusive_group()
    cache.add_argument("--use-cache", dest="use_cache", action="store_true",
                       help="Read and write the shared preprocessing cache")
    cache.add_argument("--no-use-cache", "--no-cache", dest="use_cache", action="store_false",
                       help="Recompute preprocessing and bypass the shared cache")
    result.set_defaults(use_cache=None)
    return result


def config_from(args, dataset_key="data.datasets"):
    config = load_config(args.config, args.overrides)
    changed = False
    if args.datasets is not None:
        value = normalize_datasets(args.datasets)
        if dataset_key == "data.datasets":
            config.data.datasets = value
        elif dataset_key == "evaluation.datasets":
            config.evaluation.datasets = value
        else:
            raise ValueError(f"Unknown dataset config key: {dataset_key}")
        changed = True
    if args.use_cache is not None:
        # Keep the existing cache contract field so toggling cache usage does not
        # change the validated preprocessing-version hash.
        config.preprocessing.cache_enabled = args.use_cache
        changed = True
    if changed:
        validate(config)
    return config
