"""Refuse GREEN launch until every earlier requested stage is validated."""
from .common import ROOT,OUT,INDEX,read,write
from src.evaluation.clinical import file_digest


def main():
    sig=read(OUT/'significance/completion.json')
    assert sig['status']=='complete' and sig['completed']==1050
    assert file_digest(sig['workbook']['path'])==sig['workbook']['sha256']
    source=read(INDEX)
    assert file_digest(source['master'])==source['master_sha256']
    protected=read(OUT/'protected_source_hashes.json')
    for name,sha in protected.items():
        assert file_digest(ROOT/name)==sha,('Protected original code/config changed',name)
    smoke=read(OUT/'visual/rg__A0__r32/smoke.json')
    assert smoke['status']=='passed' and smoke['zero_and_text_equal_production_controls']
    plan=read(OUT/'visual/plan.json')
    validated=read(OUT/'visual/inference_validation.json')
    assert validated['status']=='passed' and validated['rows']==26880
    assert validated['plan_sha256']==file_digest(OUT/'visual/plan.json')
    for key in plan['settings']:
        folder=OUT/'visual'/key.replace('/','__');done=read(folder/'complete.json')
        assert done['status']=='complete'
        for name,sha in done['outputs'].items():assert file_digest(folder/name)==sha
        checks=read(folder/'swap_equivalence_checks.json')
        assert len(checks)==4 and all(r['same_generated_tokens'] for r in checks)
    scoring=OUT/'visual/scoring'
    assert read(scoring/'graph_complete.json')['status']=='complete'
    assert read(scoring/'clinical_verified.json')['status']=='verified'
    exported=read(OUT/'visual/export.json')
    assert exported['status'] in ('pending_green','complete')
    assert file_digest(exported['path'])==exported['sha256']
    write(scoring/'green_gate_passed.json',dict(status='passed',verified_settings=len(plan['settings']),
          graph_pairs=read(scoring/'graph_complete.json')['pairs'],pvalue_tests=sig['completed'],
          protected_source_files_unchanged=len(protected)))
    print('GREEN_GATE_PASSED',flush=True)


if __name__=='__main__':main()
