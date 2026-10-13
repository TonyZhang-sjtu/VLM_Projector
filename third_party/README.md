# Pinned upstream dependencies

`sources.lock.json` records the source URLs and exact commits used on the research server.
Source clones, installed packages, wheels and model weights are excluded from Git.

From the activated `projector` environment, run `python scripts/setup_sources.py`
to restore NeuroVFM and the FlashAttention source used for fused-dense compilation.
Use `--include-3dino` to also retrieve 3DINO source. This does not install a 3DINO
adapter into the training pipeline or download any weights.

`bash scripts/setup_projector.sh` installs the CUDA extensions and editable project
packages after `environments.yaml` has been installed. A matching CUDA toolkit and
C++ compiler are required for fused-dense compilation on a new GPU/server.

Clinical scorer source and packages are installed separately by
`python scripts/setup_clinical_metrics.py`. Their upstream revisions match the lock;
the local overlay prevents their NumPy requirements from changing training packages.
Weights are retrieved separately and remain under ignored `assets/`.

Each upstream dependency keeps its own license in its restored checkout. This
repository does not redistribute upstream source, gated weights or medical datasets.
The projector variants are controlled mechanism baselines, not full reproductions
of the cited papers; see `Pipeline_0929/PROJECTOR_ROUTES_AND_BASELINES.md`.
