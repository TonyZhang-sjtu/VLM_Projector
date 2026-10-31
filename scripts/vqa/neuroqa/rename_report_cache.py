#!/usr/bin/env python3
"""Idempotent naming migration; never modify tensor keys or transform versions."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
VERSION = '2c90586eacc0646f9bf4670fa85fce866db8a4874b88e4e47926ed2cfe1532b9'


def main():
    root = ROOT / 'assets/preprocessing'
    old, new = root / VERSION, root / 'report_generation_native1mm_v2'
    if old.is_symlink():
        if old.resolve() != new.resolve() or not new.is_dir():
            raise ValueError('Unexpected compatibility link')
    else:
        if not old.is_dir() or new.exists():
            raise ValueError('Missing source or conflicting destination; refusing overwrite')
        old.rename(new)
        try:
            old.symlink_to(new.name, target_is_directory=True)
        except Exception:
            new.rename(old)
            raise
    count = sum(1 for _ in new.glob('*/*.pt'))
    result = {'report_cache': str(new), 'version': VERSION, 'tensors': count,
              'compatibility_link': str(old), 'reason': 'old configs/checkpoints retain the same content contract'}
    (root / 'cache_names.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
