"""Prepare a balanced training window before any gradient has been computed."""
from pathlib import Path
import json
import pickle
import random


def _case_data_error(exc, sample, preprocessor):
    """Recognize only failures attributable to this case or its patch cache."""
    message = str(exc)
    lowered = message.lower()
    if any(x in lowered for x in ('out of memory', 'cannot allocate memory', "can't allocate memory")):
        return False
    if isinstance(exc, FileNotFoundError):
        filename = getattr(exc, 'filename', None)
        if not filename:
            return False
        path = Path(filename).resolve()
        roots = [Path(sample.image).parent.resolve()]
        if hasattr(preprocessor, 'cache'):
            roots.append(Path(preprocessor.cache.root).resolve())
        return any(path.is_relative_to(root) for root in roots)
    if isinstance(exc, json.JSONDecodeError):
        # __call__ reads the per-case completion record, not global protocol.
        return True
    if isinstance(exc, ValueError):
        return message.startswith(('Changed preprocessing artifact:', 'Corrupt cache identity:',
                                   'NeuroQA v2 spatial provenance fingerprint mismatch',
                                   'Image/mask do not share the expected MNI 1 mm canvas',
                                   'Invalid NeuroQA normalized intensity range',
                                   'Nonfinite patch data', 'Invalid NeuroVFM patch/coordinate shapes',
                                   'Missing/reordered grid coordinates'))
    # These are standard torch cache deserialization errors, occurring only
    # inside the per-case preprocessor call, not model forward/backward.
    if isinstance(exc, (EOFError, pickle.UnpicklingError)):
        return True
    if isinstance(exc, RuntimeError):
        return any(term in lowered for term in ('pytorchstreamreader failed', 'invalid magic number',
                                                'failed finding central directory'))
    return False


def prepare_window(window, candidates, preprocessor, *, seed, epoch, window_index, skip_invalid, invalid_cases):
    """Load all views and deterministically replace broken cases by same-sequence views.

    The mutable invalid_cases mapping is part of the trainer checkpoint. Replacement
    may repeat valid QAs and is recorded. Non-local/model/resource errors propagate.
    """
    prepared, used, events = [], set(), []
    available = {seq: sorted((s for s in candidates if s.sequence == seq), key=lambda s: s.id)
                 for seq in ('T1W', 'T2W')}
    rng = random.Random(f'{seed}:{epoch}:{window_index}')
    for seq in available:
        rng.shuffle(available[seq])
    for original in window:
        # Prefer a new QA; if the population is very small, repeat a valid
        # same-sequence QA rather than stop or change the modality quota.
        pool = available[original.sequence]
        options = ([original] + [s for s in pool if s.qa_id not in used] +
                   [s for s in pool if s.qa_id in used])
        seen = set()
        chosen = None
        for sample in options:
            if sample.id in seen or sample.case_id in invalid_cases:
                continue
            seen.add(sample.id)
            try:
                volume = preprocessor(sample)
            except Exception as exc:
                if not skip_invalid or not _case_data_error(exc, sample, preprocessor):
                    raise
                detail = {'case_id': sample.case_id, 'id': sample.id, 'qa_id': sample.qa_id,
                          'image': sample.image, 'error_type': type(exc).__name__, 'error': str(exc)}
                invalid_cases[sample.case_id] = detail
                events.append({'kind': 'invalid_case', **detail})
                continue
            chosen = (sample, volume)
            break
        if chosen is None:
            raise ValueError(f'No valid {original.sequence} replacement for {original.id}')
        sample, volume = chosen
        repeated_in_window = sample.qa_id in used
        used.add(sample.qa_id)
        prepared.append({'sample': sample, 'volume': volume})
        if sample.id != original.id:
            events.append({'kind': 'replacement', 'original': original.id, 'replacement': sample.id,
                           'original_case': original.case_id, 'replacement_case': sample.case_id,
                           'sequence': original.sequence, 'repeated_in_window': repeated_in_window})
    # A later view can expose damage in a case already loaded earlier in the
    # window. Rebuild before backward so all its views are consistently skipped.
    if any(item['sample'].case_id in invalid_cases for item in prepared):
        prepared, more = prepare_window(window, candidates, preprocessor, seed=seed, epoch=epoch,
                                        window_index=window_index, skip_invalid=skip_invalid,
                                        invalid_cases=invalid_cases)
        events.extend(more)
    return prepared, events
