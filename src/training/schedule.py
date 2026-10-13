"""Optimizer-update schedule and validation-based stopping for continuous LoRA."""
from dataclasses import asdict, dataclass
import math


PHASES = ('projector_warmup', 'joint_warmup', 'joint_cosine')


@dataclass(frozen=True)
class JointSchedule:
    total_steps: int
    projector_steps: int
    joint_steps: int
    projector_source: str = 'steps'
    joint_source: str = 'steps'

    def __post_init__(self):
        if min(self.projector_steps, self.joint_steps, self.cosine_steps) < 2:
            raise ValueError('Each joint_lora phase needs at least 2 optimizer updates (including endpoints)')

    @property
    def cosine_steps(self):
        return self.total_steps - self.projector_steps - self.joint_steps

    @property
    def cosine_start(self):
        return self.projector_steps + self.joint_steps

    def phase(self, completed_steps):
        # The phase of the next update; completed runs retain the final phase.
        if completed_steps < self.projector_steps:
            return PHASES[0]
        if completed_steps < self.cosine_start:
            return PHASES[1]
        return PHASES[2]

    def scale(self, step, component):
        step = min(max(step, 0), self.total_steps - 1)
        if step < self.projector_steps:
            return step / (self.projector_steps - 1) if component == 'projector' else 0.0
        if step < self.cosine_start:
            return (step - self.projector_steps) / (self.joint_steps - 1)
        progress = (step - self.cosine_start) / (self.cosine_steps - 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    def to_dict(self):
        return {**asdict(self), 'cosine_steps': self.cosine_steps,
                'step_unit': 'optimizer_update', 'endpoints': 'inclusive'}

    @classmethod
    def from_dict(cls, value):
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})


def resolve_joint_schedule(training, total_steps):
    lengths, sources = {}, {}
    for name in ('projector', 'joint'):
        explicit = training.get(f'{name}_warmup_steps')
        lengths[name] = explicit if explicit is not None else math.floor(
            training[f'{name}_warmup_ratio'] * total_steps + 0.5)
        sources[name] = 'steps' if explicit is not None else 'ratio'
    return JointSchedule(total_steps, lengths['projector'], lengths['joint'],
                         sources['projector'], sources['joint'])


def update_joint_validation(state, value, *, eligible, min_delta, patience):
    """Best checkpoint and cumulative meaningful improvement are independent."""
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite validation CE')
    improved = value < state['best_val_ce']
    if improved:
        state['best_val_ce'] = value
    if eligible:
        reference = state.get('early_stop_reference')
        if reference is None or value < reference - min_delta:
            state['early_stop_reference'] = value
            state['bad_epochs'] = 0
        else:
            state['bad_epochs'] += 1
        if state['bad_epochs'] >= patience:
            state['stop_reason'] = 'early_stopping'
            state['stop_step'] = state['global_step']
    return improved
