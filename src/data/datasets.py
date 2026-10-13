"""Canonical dataset scopes for the RadGenome-Brain source cohorts."""

from collections.abc import Mapping

DATASET_ORDER = ("brats", "isles", "wmh")
ALL_DATASETS = list(DATASET_ORDER)

_ALIASES = {
    "brats": "brats",
    "brats_gli": "brats",
    "brats_men": "brats",
    "brats_met": "brats",
    "isles": "isles",
    "isles22": "isles",
    "wmh": "wmh",
}


def normalize_datasets(value, *, default=None):
    """Return a canonical list; ``all`` is accepted only by itself."""
    if value is None:
        if default is None:
            raise ValueError("A dataset selection is required")
        value = default
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Mapping):
        values = None
    else:
        try:
            values = list(value)
        except TypeError:
            values = None
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError("Dataset selection must be a nonempty name or list of names")
    cleaned = [item.strip().lower() for item in values]
    if "all" in cleaned:
        if cleaned != ["all"]:
            raise ValueError("'all' cannot be combined with individual datasets")
        return list(ALL_DATASETS)
    selected = []
    for item in cleaned:
        if item not in _ALIASES:
            raise ValueError(f"Unknown RadGenome dataset {item!r}; choose brats, isles, wmh, or all")
        canonical = _ALIASES[item]
        if canonical in selected:
            raise ValueError(f"Dataset selected more than once: {item}")
        selected.append(canonical)
    return [name for name in DATASET_ORDER if name in selected]


def dataset_for_cohort(cohort):
    if cohort in ("BraTS_GLI", "BraTS_MEN", "BraTS_MET"):
        return "brats"
    if cohort == "ISLES22":
        return "isles"
    if cohort == "WMH":
        return "wmh"
    raise ValueError(f"Unknown RadGenome cohort: {cohort}")
