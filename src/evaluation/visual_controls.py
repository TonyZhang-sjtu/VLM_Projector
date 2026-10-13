import numpy as np


def shuffled_donors(samples,seed=42):
    """Fixed cyclic derangement within acquisition sequence; no target-dependent filtering."""
    groups={}
    for i,s in enumerate(samples):groups.setdefault(s.sequence,[]).append(i)
    donors={};rng=np.random.default_rng(seed)
    for sequence,indices in groups.items():
        if len(indices)<2:raise ValueError(f'Shuffle requires >=2 examples per sequence: {sequence}')
        order=rng.permutation(indices).tolist()
        donors.update({a:b for a,b in zip(order,order[1:]+order[:1])})
    return [donors[i] for i in range(len(samples))]
