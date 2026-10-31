"""Answer-only VQA: visual encoder/projector + frozen causal LLM."""
from src.config import digest
from src.data.neuroqa_manifest import target

SYSTEM = 'You answer questions about a supplied 3D brain MRI. Use the supplied image and acquisition information.'
USER = '<image>\nThe input is one 3D brain MRI volume.\nSequence: {sequence}.\n\nQuestion: {question}\n{instruction}'
INSTRUCTIONS = {
    'YN': 'Return only Yes or No.',
    'MCQ': 'Options:\n{options}\nReturn only the letter of the correct option.',
    'Open': 'Answer concisely in English. Use only information supported by this sequence.'}


class NeuroQA:
    version = 'neuroqa_single_volume_v1'
    fingerprint = digest({'system': SYSTEM, 'user': USER, 'instructions': INSTRUCTIONS,
                          'sequence_names': {'T1W': 'T1-weighted, non-contrast', 'T2W': 'T2-weighted'}})

    def messages(self, sample):
        row = sample.row
        instruction = INSTRUCTIONS[row['question_type']]
        if row['question_type'] == 'MCQ':
            options = '\n'.join(f'{chr(65+i)}. {s}' for i, s in enumerate(row['options']))
            if '<image>' in options or '<|' in options:
                raise ValueError('Reserved chat marker in MCQ choices')
            instruction = instruction.format(options=options)
        return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': USER.format(
            sequence={'T1W': 'T1-weighted, non-contrast', 'T2W': 'T2-weighted'}[sample.sequence],
            question=row['question'], instruction=instruction)}]

    def target(self, sample):
        return target(sample.row)

    def generation_budget(self, sample):
        return 8 if sample.row['question_type'] in ('MCQ', 'YN') else 192
