import json
from pathlib import Path

from src.evaluation.clinical import file_digest

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'results/anchorread3d_analysis_1108/artifact_reconciliation_1108.json'
OUT = ROOT / 'results/anchorread3d_final_eval_1108'
BASELINES = ['AvgPool3D', 'Concat3D', 'ConvPool3D', 'LocalPacker3D',
             'Perceiver_NV', 'Adaptive_Merge3D', 'Token_Select3D']
VISUAL_METHODS = ['A0', 'AvgPool3D', 'Perceiver_NV', 'Token_Select3D']


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def entries():
    index = read(INDEX)
    assert file_digest(index['master']) == index['master_sha256'], '1108 workbook changed'
    allowed = set(BASELINES + ['A0', 'A1', 'A2', 'A3', 'A4'])
    result = {e['key']: e for e in index['paper_settings'] if e['key'].split('/')[1] in allowed}
    assert len(result) == 112
    return index, result
