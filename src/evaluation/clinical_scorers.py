"""Thin local-device adapters around pinned official report metric code."""
import gc
import json
import re
import tarfile
from pathlib import Path
from .clinical import file_digest, write_json


def failure(status, details):
    return {'status': status, 'values': {}, 'details': details}


def green_structure_error(text):
    """Official parser maps absent/malformed fields to zero; do not silently accept that."""
    for category in ('Clinically Significant Errors', 'Matched Findings'):
        matches = re.findall(rf'\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)', text, re.DOTALL)
        if len(matches) != 1:
            return f'Missing/duplicate {category} section'
        section = matches[0].strip()
        if section.startswith('No'):
            continue
        if category == 'Matched Findings':
            if not re.match(r'\d+\.', section):
                return 'Malformed matched-finding count'
        else:
            lines = re.findall(r'\(([a-f1-6])\) .*', section)
            if not lines or len(lines) != len(set(lines)):
                return 'Missing/duplicate error category counts'
            if len(re.findall(r'\([a-f1-6]\) .*?: \d+\.', section)) != len(lines):
                return 'Malformed error count'
    return None


class BaseScorer:
    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.impl = None

    def memory(self):
        import torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device)}
        return {}

    def close(self):
        import torch
        self.impl = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


class GreenScorer(BaseScorer):
    def __init__(self, assets, device, max_length=2048):
        super().__init__(device)
        from green_score import GREEN
        # cpu=True bypasses the upstream auto-DDP branch on multi-GPU workstations.
        # Move only this scoring model to the explicitly selected device afterward.
        self.impl = GREEN(str(Path(assets) / 'green'), cpu=True, compute_summary_stats=False)
        self.impl.model.to(self.device)
        self.max_length = max_length

    def score(self, reference, prediction):
        import torch
        from green_score.utils import make_prompt, clean_responses
        tok = self.impl.tokenizer
        prompt = make_prompt(reference, prediction)
        chat = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': ''}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        batch = tok([text], return_tensors='pt', padding=True, truncation=False).to(self.device)
        length = batch['input_ids'].shape[1]
        if length >= self.max_length:
            return failure('input_too_long', {'input_tokens': length, 'max_length': self.max_length})
        with torch.inference_mode():
            outputs = self.impl.model.generate(
                **batch, max_length=self.max_length, do_sample=False, temperature=None, top_p=None,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        analysis = clean_responses(tok.batch_decode(outputs, skip_special_tokens=True)[0])
        details = {'analysis': analysis, 'input_tokens': length,
                   'generated_tokens': outputs.shape[1] - length, **self.memory()}
        if outputs.shape[1] >= self.max_length and outputs[0, -1].item() != tok.eos_token_id:
            return failure('output_truncated', details)
        problem = green_structure_error(analysis)
        if problem:
            return failure('parse_error', dict(details, error=problem))
        counts = self.impl.compute_error_count(analysis)
        value = self.impl.compute_green(analysis)
        details.update(matched_findings=counts[-1], significant_errors=counts[:6],
                       error_categories=self.impl.sub_categories)
        if value is None:
            return failure('parse_error', details)
        return {'status': 'ok', 'values': {'green': float(value)}, 'details': details}


def radgraph_runtime(assets):
    """Extract verified weights and substitute local base-model paths only."""
    from filelock import FileLock
    assets = Path(assets).resolve()
    cache = assets / 'radgraph/runtime'
    target = cache / 'radgraph-xl'
    with FileLock(str(assets / 'radgraph/runtime.lock')):
        # Verify the actual extracted runtime files too: a modified weights.th must
        # not be scored under the pristine archive's cache fingerprint.
        import hashlib
        import shutil
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(assets / 'radgraph/radgraph-xl.tar.gz') as archive:
            for member in archive.getmembers():
                path = target / member.name
                if not path.resolve().is_relative_to(target.resolve()) or member.issym() or member.islnk():
                    raise ValueError('Unsafe RadGraph archive entry')
                if not member.isfile() or member.name.lstrip('./') == 'config.json':
                    continue
                checksum = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        checksum.update(block)
                if not path.exists() or file_digest(path) != checksum.hexdigest():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp = path.with_name(path.name + '.tmp')
                    with archive.extractfile(member) as source, temp.open('wb') as dest:
                        shutil.copyfileobj(source, dest)
                    temp.replace(path)
        # Read the pristine archived config on each run, never a previously mutated config.
        with tarfile.open(assets / 'radgraph/radgraph-xl.tar.gz') as archive:
            member = next(m for m in archive.getmembers() if m.name.lstrip('./') == 'config.json')
            config = json.load(archive.extractfile(member))
        base = str(assets / 'cxr-bert')
        config['dataset_reader']['token_indexers']['bert']['model_name'] = base
        config['model']['embedder']['token_embedders']['bert']['model_name'] = base
        write_json(target / 'config.json', config)
    return cache


class RadGraphScorer(BaseScorer):
    def __init__(self, assets, device):
        super().__init__(device)
        from radgraph import F1RadGraph
        self.impl = F1RadGraph(model_type='radgraph-xl', reward_level='all',
                               cuda=self.device.index if self.device.type == 'cuda' else -1,
                               model_cache_dir=str(radgraph_runtime(assets)),
                               tokenizer_cache_dir=str(Path(assets) / 'hub'))

    def score(self, reference, prediction):
        import torch
        with torch.inference_mode():
            _, scores, hyps, refs = self.impl(refs=[reference], hyps=[prediction])
        if len(hyps) != 1 or len(refs) != 1:
            raise ValueError('RadGraph graph alignment mismatch')
        keys = ('radgraph_simple', 'radgraph_partial', 'radgraph_complete')
        details = {'prediction_graph': hyps[0], 'reference_graph': refs[0], **self.memory()}
        if not hyps[0]['entities'] or not refs[0]['entities']:
            details['warnings'] = ['empty_entities']
        return {'status': 'ok', 'values': {k: float(s[0]) for k, s in zip(keys, scores)}, 'details': details}

    def close(self):
        from radgraph.allennlp.common import cached_transformers
        cached_transformers._model_cache.clear()
        super().close()


class RateScorer(BaseScorer):
    def __init__(self, assets, device):
        super().__init__(device)
        from RaTEScore import RaTEScore
        self.impl = RaTEScore(bert_model=str(Path(assets) / 'rate-ner'),
                              eval_model=str(Path(assets) / 'biolord'), batch_size=1,
                              use_gpu=self.device.type == 'cuda', affinity_matrix='long')

    def entities(self, text):
        from RaTEScore.score import run_ner
        from RaTEScore.utils import sentence_split
        sentences, _ = sentence_split([text])
        if not sentences:
            return [], []
        lengths = [len(ids) for ids in self.impl.tokenizer(sentences, truncation=False)['input_ids']]
        if max(lengths) > 512:
            raise ValueError('RaTE sentence exceeds 512 tokens; refusing silent report truncation')
        result = run_ner([text], self.impl.idx2label, self.impl.tokenizer,
                         self.impl.model, self.impl.device, 1)
        if len(result) != 1:
            raise ValueError('RaTE entity/report alignment mismatch')
        return result[0], lengths

    def score(self, reference, prediction):
        from RaTEScore.score import process_embedding
        from RaTEScore.utils import compute
        refs, ref_lengths = self.entities(reference)
        hyps, hyp_lengths = self.entities(prediction)
        a, a_types = process_embedding(refs, self.impl.eval_tokenizer, self.impl.eval_model, self.impl.device)
        b, b_types = process_embedding(hyps, self.impl.eval_tokenizer, self.impl.eval_model, self.impl.device)
        details = {'reference_entities': refs, 'prediction_entities': hyps,
                   'reference_sentence_tokens': ref_lengths, 'prediction_sentence_tokens': hyp_lengths,
                   'affinity_matrix': 'long', **self.memory()}
        warnings = []
        if not len(a) or not len(b):
            value = 0.5  # Official no-entity fallback, explicitly reported as such.
            warnings.append('empty_entities_official_0.5')
        else:
            forward = compute(a, b, a_types, b_types, self.impl.affinity_matrix)
            backward = compute(b, a, b_types, a_types, self.impl.affinity_matrix)
            value = 2 * forward * backward / (forward + backward) if forward + backward else 0
            details.update(reference_to_prediction=forward, prediction_to_reference=backward)
        entities = [pair[0] for pair in refs + hyps]
        if entities and any(len(t) > 30 for t in self.impl.eval_tokenizer(entities, truncation=False)['input_ids']):
            warnings.append('entity_embedding_uses_official_30_token_limit')
        details['warnings'] = warnings
        return {'status': 'ok', 'values': {'rate': float(value)}, 'details': details}


class BertScoreScorer(BaseScorer):
    """Official token matching with explicit reference/candidate orientation."""
    def __init__(self, assets, device):
        super().__init__(device)
        from bert_score import BERTScorer
        from .clinical_runtime import BERTSCORE_SETTINGS
        settings = BERTSCORE_SETTINGS
        self.impl = BERTScorer(
            model_type=str(Path(assets).resolve() / 'bert-score'),
            num_layers=settings['num_layers'], lang=settings['lang'],
            idf=settings['idf'], rescale_with_baseline=settings['rescale_with_baseline'],
            use_fast_tokenizer=settings['use_fast_tokenizer'], all_layers=settings['all_layers'],
            device=str(self.device), batch_size=1,
        )
        self.impl._model.float().eval()
        self.max_tokens = min(settings['max_tokens'], self.impl._tokenizer.model_max_length,
                              self.impl._model.config.max_position_embeddings)

    def score(self, reference, prediction):
        import torch
        if not reference.strip() or not prediction.strip():
            return failure('empty_reference' if not reference.strip() else 'empty_prediction', {})
        # Upstream sent_encode strips edges and truncates silently. Count with
        # the exact same slow tokenizer and special tokens, WITHOUT truncation.
        lengths = {f'{side}_tokens': len(self.impl._tokenizer.encode(
            text.strip(), add_special_tokens=True, truncation=False))
            for side, text in (('reference', reference), ('prediction', prediction))}
        details = dict(lengths, max_tokens=self.max_tokens, num_layers=self.impl.num_layers,
                       official_hash=self.impl.hash, truncated=False)
        if max(lengths.values()) > self.max_tokens:
            return failure('input_too_long', details)
        with torch.inference_mode():
            precision, recall, f1 = self.impl.score(cands=[prediction], refs=[reference], batch_size=1)
        details.update(self.memory())
        return {'status': 'ok', 'values': {
            'bertscore_precision': float(precision[0]), 'bertscore_recall': float(recall[0]),
            'bertscore_f1': float(f1[0]),
        }, 'details': details}


def make_scorer(metric, assets, device, green_max_length=2048):
    if metric == 'green':
        return GreenScorer(assets, device, green_max_length)
    if metric == 'radgraph':
        return RadGraphScorer(assets, device)
    if metric == 'rate':
        return RateScorer(assets, device)
    if metric == 'bertscore':
        return BertScoreScorer(assets, device)
    raise ValueError(metric)
