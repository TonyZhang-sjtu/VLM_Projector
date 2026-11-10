import json
import sys

import pytest

from .clinical_worker import supervise


def test_scorer_runtime_error_stops_only_its_own_child(tmp_path):
    cache=tmp_path/'cache';(cache/'green').mkdir(parents=True)
    output=cache/'green'/'example.json';marker=tmp_path/'continued_after_error'
    program=("import json,time;from pathlib import Path;"
             f"Path({str(output)!r}).write_text(json.dumps(dict(status='scoring_error')));"
             f"time.sleep(2);Path({str(marker)!r}).write_text('should not happen')")
    status=tmp_path/'status.json'
    with pytest.raises(RuntimeError,match='runtime error'):
        supervise([sys.executable,'-c',program],cache,status,interval=.01,stall_seconds=5)
    assert not marker.exists()
    result=json.loads(status.read_text())
    assert result['status']=='failed' and result['returncode']!=0


def test_known_green_format_failure_is_reported_without_killing_job(tmp_path):
    cache=tmp_path/'cache';(cache/'green').mkdir(parents=True)
    output=cache/'green'/'example.json'
    program=("import json;from pathlib import Path;"
             f"Path({str(output)!r}).write_text(json.dumps(dict(status='format_repair_failed')))")
    status=tmp_path/'status.json'
    supervise([sys.executable,'-c',program],cache,status,interval=.01,stall_seconds=5)
    result=json.loads(status.read_text())
    assert result['status']=='complete'
    assert result['cache_counts']['green']=={'format_repair_failed':1}
