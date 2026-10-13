"""Per-volume token selection hook. The default preserves the complete grid.

Call after tokenization and before batching. A future image-derived selector
must return a boolean keep mask, without using ground-truth lesion masks.
The caller must rebuild packed sequence lengths after any custom selection.
This module does not change NeuroVFM's own preprocessing defaults.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class TokenSelection:
    tokens: Tensor
    coords: Tensor
    kept_indices: Tensor
    original_count: int
    policy: str


def select_tokens(
    tokens: Tensor,
    coords: Tensor,
    *,
    policy: str = "none",
    selector: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> TokenSelection:
    """Keep token values, spatial coordinates and original indices aligned.

    ``tokens`` is [N, ...], ``coords`` is [N, 3], for one volume only.
    ``none`` returns the original tensors with identity indices. ``custom``
    is reserved for an explicitly supplied foreground-selection function.
    No pooling, token merging or detached copies are performed.
    """
    if tokens.ndim < 2 or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Expected tokens [N, ...] and coords [N, 3].")
    count = tokens.shape[0]
    if count == 0 or coords.shape[0] != count:
        raise ValueError("A nonempty, matching token/coordinate grid is required.")
    if tokens.device != coords.device:
        raise ValueError("Tokens and coordinates must be on the same device.")
    indices = torch.arange(count, device=tokens.device)
    if policy == "none":
        if selector is not None:
            raise ValueError("A selector requires explicit policy='custom'.")
        return TokenSelection(tokens, coords, indices, count, policy)
    if policy != "custom" or selector is None:
        raise ValueError("Use policy='none' or supply a selector with 'custom'.")
    keep = selector(tokens, coords)
    if not isinstance(keep, Tensor) or keep.dtype != torch.bool:
        raise ValueError("The selector must return a boolean torch tensor.")
    if keep.shape != (count,) or keep.device != tokens.device:
        raise ValueError("The keep mask must have shape [N] on the token device.")
    if not bool(keep.any()):
        raise ValueError("Selection must retain at least one token.")
    return TokenSelection(tokens[keep], coords[keep], indices[keep], count, policy)
