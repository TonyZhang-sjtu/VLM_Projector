import os
from pathlib import Path
import tempfile
import torch


def atomic_torch_save(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix='.partial-')
    os.close(fd)
    try:
        with open(temporary, 'wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TensorCache:
    def __init__(self, root):
        # Keep content/version identity intact while using human-readable stores.
        # resolve() follows the migration compatibility link for old checkpoints
        # and already-running consumers; no fingerprinted transform code changes.
        self.root = Path(root).resolve()

    def get(self, key):
        path = self.root / key[:2] / f'{key}.pt'
        if not path.exists():
            return None
        value = torch.load(path, map_location='cpu', weights_only=True)
        if value['key'] != key:
            raise ValueError(f'Corrupt cache identity: {path}')
        return value['value']

    def put(self, key, value):
        atomic_torch_save({'key': key, 'value': value}, self.root / key[:2] / f'{key}.pt')
