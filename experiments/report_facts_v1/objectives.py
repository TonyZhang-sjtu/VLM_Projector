"""Final-Z fact heads and differentiable frozen-decoder answer pairing."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from src.data.sequence_mapping import SEQUENCE_NAMES
from src.models.decoders.llama import LlamaDecoder
from .schema import FIELDS


class FactHeads(nn.Module):
    def __init__(self,width,fields):
        super().__init__()
        self.heads=nn.ModuleDict({field:nn.Sequential(nn.LayerNorm(width),nn.Linear(width,len(FIELDS[field]['values'])))
                                 for field in fields})

    def objective(self,z,allocation,facts):
        local=z[allocation['global_tokens']:]
        if len(local)!=allocation['num_slots']*4:
            raise ValueError('Fact heads require exactly four final local tokens per group')
        pooled=local.reshape(allocation['num_slots'],4,z.shape[-1]).mean(0)
        losses=[];correct=z.new_zeros(())
        for field,head in self.heads.items():
            if field not in facts:continue
            logits=head(pooled[FIELDS[field]['role']]).float()
            target=FIELDS[field]['values'].index(facts[field]['value'])
            losses.append(F.cross_entropy(logits[None],torch.tensor([target],device=z.device)))
            correct=correct+(logits.argmax()==target).detach()
        # Keep unused heads connected in partial-label batches, including DDP.
        zero=sum((p.sum()*0 for p in self.parameters()),z.sum()*0)
        return (torch.stack(losses).mean()+zero if losses else zero),len(losses),correct


@dataclass(frozen=True)
class FactQuestion:
    id: str
    sequence: str
    field: str
    answer: str


class FactTask:
    def messages(self,sample):
        return [dict(role='system',content='Describe the supplied brain MRI image. Answer the question using the image and acquisition information.'),
                dict(role='user',content=f'<image>\nSequence: {SEQUENCE_NAMES[sample.sequence]}.\n{FIELDS[sample.field]["question"]}')]

    def target(self,sample):
        return sample.answer


class FactDecoderView:
    """Borrow frozen decoder methods without mutating its report task or state.

    The answer appears exclusively after the assistant boundary. It is a teacher
    forcing target and is never forwarded to the encoder/projector/prompt.
    """
    task=FactTask()
    text_ids=LlamaDecoder.text_ids
    prepare=LlamaDecoder.prepare

    def __init__(self,decoder):
        self.decoder=decoder

    def __getattr__(self,name):
        return getattr(self.decoder,name)

    def score(self,z,sample,field,value):
        spec=FIELDS[field]
        question=FactQuestion(sample.id,sample.sequence,field,spec['answers'][spec['values'].index(value)])
        batch=self.prepare([question],[z],training=True)
        loss,count=self.decoder.loss(batch)
        return -loss/count


def pairing_loss(positive_score,negative_score,margin,positive_ce_weight):
    ranking=F.softplus(margin-positive_score+negative_score)
    positive_ce=-positive_score
    return ranking+positive_ce_weight*positive_ce,ranking,positive_ce
