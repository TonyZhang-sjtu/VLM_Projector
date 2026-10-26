"""Report-task token scoring with exact chat, character and causal alignment."""
from dataclasses import replace
import torch
from torch.nn import functional as F


def answer_masks(decoder, sample, spans):
    tokenizer = decoder.tokenizer
    if not tokenizer.is_fast:
        raise ValueError('Report span supervision requires verified fast-tokenizer offsets')
    messages = decoder.task.messages(sample)
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rendered = tokenizer.apply_chat_template(
        messages + [dict(role='assistant', content=sample.finding)],
        tokenize=False, add_generation_prompt=False)
    # The installed Llama template applies Jinja `trim` to message content.
    # Account only for this verified transformation, never guess an alignment.
    target = sample.finding.strip()
    leading = len(sample.finding)-len(sample.finding.lstrip())
    if not rendered.startswith(prefix + target):
        raise ValueError('Chat rendering altered report text beyond whitespace trim')
    origin = len(prefix)-leading
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, boundary = decoder.text_ids(sample, True)
    if encoded['input_ids'] != ids:
        raise ValueError('Offset-tokenization differs from actual decoder token IDs')
    offsets = encoded['offset_mapping'][boundary:]
    result = {}
    for field, values in spans.items():
        mask = [False] * len(offsets)
        for span in values:
            a, b = span['start'], span['end']
            if not 0 <= a < b <= len(sample.finding) or sample.finding[a:b] != span['text']:
                raise ValueError(f'{sample.id}: stale/misaligned {field} evidence span')
            matched = []
            for i, (start, end) in enumerate(offsets):
                if start < origin+b and origin+a < end:
                    mask[i] = True
                    matched.append(i)
            if not matched:
                raise ValueError(f'{sample.id}: attribute span has no answer tokens')
        if not any(mask):
            raise ValueError('Empty field mask')
        # EOT is part of report CE, never part of an attribute score.
        if mask[-1]:
            raise ValueError('Attribute mask includes the chat terminator')
        result[field] = mask
    return result


def token_nll(decoder, sample, z):
    batch = decoder.prepare([sample], [z], training=True)
    hidden = decoder.lm.model(inputs_embeds=batch.inputs_embeds,
        attention_mask=batch.attention_mask, position_ids=batch.position_ids,
        use_cache=False, return_dict=True).last_hidden_state
    target = batch.labels[:, 1:]
    valid = target != -100
    selected, target = hidden[:, :-1][valid], target[valid]
    if len(target) != decoder.answer_count(sample):
        raise ValueError('Answer count/causal shift mismatch')
    values = []
    chunk = decoder.config.decoder.ce_chunk_size
    for start in range(0, len(target), chunk):
        logits = decoder.lm.lm_head(selected[start:start+chunk]).float()
        values.append(F.cross_entropy(logits, target[start:start+chunk], reduction='none'))
    return torch.cat(values), batch.lengths[0]


def field_losses(nll, masks):
    return {field: nll[torch.tensor(mask, dtype=torch.bool, device=nll.device)].mean()
            for field, mask in masks.items()}


def score(nll, masks):
    losses = field_losses(nll, masks)
    if not losses:
        raise ValueError('A candidate must contain a supervised attribute')
    return -torch.stack(list(losses.values())).mean()


def candidate_sample(sample, candidate):
    # Only the assistant target changes. The report prompt and image stay intact.
    return replace(sample, finding=candidate['text'])
