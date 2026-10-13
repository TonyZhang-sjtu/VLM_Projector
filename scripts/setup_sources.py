"""Restore pinned upstream source without replacing existing checkouts."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--include-3dino', action='store_true')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'third_party/sources.lock.json').read_text())
    names = {'neurovfm', 'flash-attention'}
    if args.include_3dino:
        names.add('3dino')
    for entry in lock['repositories']:
        if entry['name'] not in names:
            continue
        path = ROOT / entry['path']
        if not path.exists():
            if args.check_only:
                raise SystemExit(f'Missing source: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', entry['url'], str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', entry['commit']], check=True)
        if not (path / '.git').exists():
            raise SystemExit(f'Existing path is not an upstream checkout: {path}')
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'diff', 'HEAD', '--name-only'], text=True).strip()
        if actual != entry['commit'] or dirty:
            raise SystemExit(f'Checkout differs from lock: {path}; refusing to overwrite it')
        print(f'{entry["name"]}: {actual}')


if __name__ == '__main__':
    main()
