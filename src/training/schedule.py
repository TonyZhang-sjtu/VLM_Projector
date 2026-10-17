"""Optimizer-update schedule and validation-based stopping for continuous LoRA."""
from dataclasses import asdict, dataclass
import math


PROJECTOR_ONLY_PHASES = ('projector_warmup', 'projector_hold')
PHASES = (*PROJECTOR_ONLY_PHASES, 'joint_warmup', 'joint_cosine')


@dataclass(frozen=True)
class JointSchedule:
    total_steps: int
    projector_steps: int
    joint_steps: int
    projector_source: str = 'steps'
    joint_source: str = 'steps'
    cosine_min_lr_ratio: float = 0.0
    joint_warmup_projector_mode: str = 'restart'
    # Last optional field preserves existing positional constructors.
    projector_hold_steps: int = 0

    def __post_init__(self):
        for name in ('total_steps', 'projector_steps', 'joint_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f'{name} must be an integer of at least 2')
        if type(self.projector_hold_steps) is not int or self.projector_hold_steps < 0:
            raise ValueError('projector_hold_steps must be a nonnegative integer')
        if min(self.projector_steps, self.joint_steps, self.cosine_steps) < 2:
            raise ValueError('The total budget must cover both warm-ups, the projector hold, '
                             'and at least 2 joint cosine updates')
        if (type(self.cosine_min_lr_ratio) not in (int, float)
                or not math.isfinite(self.cosine_min_lr_ratio)
                or not 0 <= self.cosine_min_lr_ratio < 1):
            raise ValueError('cosine_min_lr_ratio must be finite and between 0 (inclusive) and 1')
        if self.joint_warmup_projector_mode not in ('restart', 'hold_peak'):
            raise ValueError('joint_warmup_projector_mode must be restart or hold_peak')

    @property
    def cosine_steps(self):
        return self.total_steps - self.cosine_start

    @property
    def joint_start(self):
        return self.projector_steps + self.projector_hold_steps

    @property
    def cosine_start(self):
        return self.joint_start + self.joint_steps

    def phase(self, completed_steps):
        # The phase of the next update; completed runs retain the final phase.
        if completed_steps < self.projector_steps:
            return 'projector_warmup'
        if completed_steps < self.joint_start:
            return 'projector_hold'
        if completed_steps < self.cosine_start:
            return 'joint_warmup'
        return 'joint_cosine'

    def scale(self, step, component):
        if component not in ('projector', 'encoder'):
            raise ValueError(f'Unknown optimizer component: {component}')
        step = min(max(step, 0), self.total_steps - 1)
        if step < self.projector_steps:
            return step / (self.projector_steps - 1) if component == 'projector' else 0.0
        if step < self.joint_start:
            return 1.0 if component == 'projector' else 0.0
        if step < self.cosine_start:
            if component == 'projector' and self.joint_warmup_projector_mode == 'hold_peak':
                return 1.0
            return (step - self.joint_start) / (self.joint_steps - 1)
        progress = (step - self.cosine_start) / (self.cosine_steps - 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.cosine_min_lr_ratio + (1 - self.cosine_min_lr_ratio) * cosine

    def to_dict(self):
        lengths = (self.projector_steps, self.projector_hold_steps,
                   self.joint_steps, self.cosine_steps)
        phase_ranges, completed = [], 0
        for phase, steps in zip(PHASES, lengths):
            if steps:
                phase_ranges.append({'phase': phase, 'steps': steps,
                                     'start_step': completed + 1, 'end_step': completed + steps})
            completed += steps
        return {**asdict(self), 'cosine_steps': self.cosine_steps,
                'phase_ranges': phase_ranges,
                'step_unit': 'optimizer_update', 'endpoints': 'inclusive'}

    @classmethod
    def from_dict(cls, value):
        # Legacy plans have no hold phase and keep their original LR defaults.
        # Derived metadata (including phase_ranges) is always recomputed.
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})


def resolve_joint_schedule(training, total_steps):
    lengths, sources = {}, {}
    for name in ('projector', 'joint'):
        explicit = training.get(f'{name}_warmup_steps')
        lengths[name] = explicit if explicit is not None else math.floor(
            training[f'{name}_warmup_ratio'] * total_steps + 0.5)
        sources[name] = 'steps' if explicit is not None else 'ratio'
    return JointSchedule(total_steps, lengths['projector'], lengths['joint'],
                         sources['projector'], sources['joint'],
                         training.get('cosine_min_lr_ratio', 0.0),
                         training.get('joint_warmup_projector_mode', 'restart'),
                         training.get('projector_hold_steps', 0))


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
