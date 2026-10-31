from dataclasses import dataclass
import json

import pytest

from src.data.neuroqa_runtime import prepare_window


@dataclass
class Sample:
    id: str
    qa_id: str
    case_id: str
    sequence: str
    image: str


def sample(tmp_path, name, seq):
    return Sample(name, name, name, seq, str(tmp_path / name / 'image.nii.gz'))


def test_missing_case_is_replaced_with_same_sequence_deterministically(tmp_path):
    one = sample(tmp_path, 'one', 'T1W')
    two = sample(tmp_path, 'two', 'T2W')
    fallback = sample(tmp_path, 'fallback', 'T1W')
    pre = lambda s: (_ for _ in ()).throw(FileNotFoundError(2, 'missing', s.image)) if s is one else s.id
    def run():
        return prepare_window([one, two], [one, two, fallback], pre, seed=42, epoch=0,
                              window_index=0, skip_invalid=True, invalid_cases={})
    items, events = run()
    assert [x['sample'].id for x in items] == ['fallback', 'two']
    assert [e['kind'] for e in events] == ['invalid_case', 'replacement']
    assert run()[1] == events


def test_model_or_resource_errors_are_not_hidden(tmp_path):
    one = sample(tmp_path, 'one', 'T1W')
    def failed(_):
        raise RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        prepare_window([one], [one], failed, seed=1, epoch=0, window_index=0,
                       skip_invalid=True, invalid_cases={})


def test_other_process_missing_file_is_not_assumed_a_case_error(tmp_path):
    one = sample(tmp_path, 'one', 'T1W')
    def failed(_):
        raise FileNotFoundError(2, 'missing', str(tmp_path / 'global_template'))
    with pytest.raises(FileNotFoundError):
        prepare_window([one], [one], failed, seed=1, epoch=0, window_index=0,
                       skip_invalid=True, invalid_cases={})


def test_later_failure_discards_earlier_view_of_same_case(tmp_path):
    a = sample(tmp_path, 'first', 'T1W')
    b = Sample('second', 'second', a.case_id, 'T2W', a.image)
    c = sample(tmp_path, 'replacement-t1', 'T1W')
    d = sample(tmp_path, 'replacement-t2', 'T2W')
    def pre(s):
        if s is b:
            raise FileNotFoundError(2, 'missing', s.image)
        return s.id
    items, events = prepare_window([a, b], [a, b, c, d], pre, seed=1, epoch=0,
                                   window_index=0, skip_invalid=True, invalid_cases={})
    assert [x['sample'].id for x in items] == [c.id, d.id]
    assert any(e['kind'] == 'invalid_case' for e in events)


def test_reuse_is_last_resort_when_no_new_qa_remains(tmp_path):
    valid = sample(tmp_path, 'valid', 'T1W')
    missing = sample(tmp_path, 'missing', 'T1W')
    def pre(s):
        if s is missing:
            raise FileNotFoundError(2, 'missing', s.image)
        return s.id
    items, events = prepare_window([valid, missing], [valid, missing], pre, seed=3, epoch=0,
                                   window_index=0, skip_invalid=True, invalid_cases={})
    assert [x['sample'].id for x in items] == [valid.id, valid.id]
    assert next(e for e in events if e['kind'] == 'replacement')['repeated_in_window']


@pytest.mark.parametrize('error', [json.JSONDecodeError('bad record', '{', 1),
                                  ValueError('Invalid NeuroQA normalized intensity range'),
                                  ValueError('Image/mask do not share the expected MNI 1 mm canvas'),
                                  ValueError('NeuroQA v2 spatial provenance fingerprint mismatch')])
def test_known_local_corruption_is_skipped(tmp_path, error):
    broken = sample(tmp_path, 'broken', 'T1W')
    healthy = sample(tmp_path, 'healthy', 'T1W')
    def pre(s):
        if s is broken:
            raise error
        return s.id
    items, events = prepare_window([broken], [broken, healthy], pre, seed=1, epoch=0,
                                   window_index=0, skip_invalid=True, invalid_cases={})
    assert items[0]['sample'] is healthy
    assert events[0]['case_id'] == broken.case_id


@pytest.mark.parametrize('message', ['NeuroQA preprocessing protocol/template mismatch',
                                     'NeuroQA registration settings/dependencies differ from the configured protocol'])
def test_global_protocol_mismatch_is_not_hidden(tmp_path, message):
    one = sample(tmp_path, 'one', 'T1W')
    def failed(_):
        raise ValueError(message)
    with pytest.raises(ValueError, match=message):
        prepare_window([one], [one], failed, seed=1, epoch=0, window_index=0,
                       skip_invalid=True, invalid_cases={})
