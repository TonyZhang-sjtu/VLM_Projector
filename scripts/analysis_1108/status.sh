#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/sda/mingjie/conda_envs/projector/bin/python}"
cd "$PROJECT_DIR"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
root=Path('results/anchorread3d_final_eval_1108')
for p in [root/'workflow_status.json', *sorted((root/'queues').glob('*/status.json'))]:
    if not p.exists(): continue
    d=json.loads(p.read_text())
    if 'completed' in d and isinstance(d['completed'],list):d['completed']=len(d['completed'])
    if isinstance(d.get('failed'),list):d['failed']=len(d['failed'])
    d.pop('gpus',None)
    print(p, json.dumps(d,ensure_ascii=False))
p=root/'monitor_status.json'
if p.exists():
    d=json.loads(p.read_text())
    print('Monitor',json.dumps({k:d.get(k) for k in ['time','alerts','resolved_alerts','unresolved_alerts','reused_ce_exports','scoring_cache_progress']},ensure_ascii=False))
PY
