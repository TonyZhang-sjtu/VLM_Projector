from src.data.sequence_mapping import SEQUENCE_NAMES
from src.config import digest

SYSTEM = 'You assist with describing brain MRI findings. Base your statements on the supplied image and acquisition information.'
USER = '''<image>
The input is one 3D brain MRI volume.
Sequence: {sequence_name}.

Write a concise, self-contained Findings paragraph in English for this sequence.
Describe visible abnormalities, including their location and laterality, signal intensity, morphology, and any evident edema or mass effect. Include any other relevant imaging observations supported by this sequence. Mention contrast enhancement only for a post-contrast input.
Do not assume access to other sequences or prior examinations. Do not invent measurements, comparisons, or unobserved findings. Avoid references such as "as above" or "the same as on T1W".
Return only the findings, without a separate impression, differential diagnosis, or recommendations.'''


class SingleSequenceFindings:
    version='single_sequence_findings_v2'
    fingerprint=digest({'system':SYSTEM,'user':USER,'sequence_names':SEQUENCE_NAMES})

    def messages(self,sample):
        return [{'role':'system','content':SYSTEM},
                {'role':'user','content':USER.format(sequence_name=SEQUENCE_NAMES[sample.sequence])}]

    def target(self,sample):
        if not sample.finding.strip() or '<|' in sample.finding or '<image>' in sample.finding:
            raise ValueError(f'Empty target or reserved chat token in target: {sample.id}')
        return sample.finding


def build_task(config):
    if config.task.name!='single_sequence_findings' or config.task.prompt_version!=SingleSequenceFindings.version:
        raise ValueError('Unknown task or prompt version')
    return SingleSequenceFindings()
