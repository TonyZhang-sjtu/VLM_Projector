from dataclasses import dataclass
from torch import Tensor, nn


@dataclass
class DecoderBatch:
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    labels: Tensor
    lengths: list
    prompt_token_counts: list


class DecoderAdapter(nn.Module):
    hidden_size: int

    def prepare(self, samples, visual, training=True) -> DecoderBatch:
        raise NotImplementedError
