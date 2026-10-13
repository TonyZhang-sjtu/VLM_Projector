"""Sequence identity comes only from source IDs, never report text."""
import re

SEQUENCE_NAMES = {
    "T1W": "T1-weighted MRI without contrast",
    "T1C": "post-contrast T1-weighted MRI",
    "T2W": "T2-weighted MRI",
    "FLAIR": "fluid-attenuated inversion recovery (FLAIR) MRI",
    "ADC": "apparent diffusion coefficient (ADC) map",
}


def sequence_of(cohort, sample_id):
    if cohort in ("BraTS_GLI", "BraTS_MEN", "BraTS_MET"):
        match = re.fullmatch(r"BraTS-(GLI|MEN|MET)-(\d+)-(\d+)-(t1n|t1c|t2w|t2f)", sample_id)
        if match and cohort == "BraTS_" + match[1]:
            return {"t1n": "T1W", "t1c": "T1C", "t2w": "T2W", "t2f": "FLAIR"}[match[4]]
    if cohort == "ISLES22" and re.fullmatch(r"sub-strokecase\d+_ses-\d+_adc", sample_id):
        return "ADC"
    if cohort == "WMH" and re.fullmatch(r".+_\d+_FLAIR", sample_id):
        return "FLAIR"
    raise ValueError(f"Unknown cohort/sequence identity: {cohort}/{sample_id}")


def patient_and_case(cohort, sample_id):
    sequence_of(cohort, sample_id)
    if cohort.startswith("BraTS_"):
        parts = sample_id.split("-")
        return "-".join(parts[:3]), "-".join(parts[:4])
    if cohort == "ISLES22":
        return "ISLES22:" + sample_id.split("_ses-")[0], sample_id.rsplit("_", 1)[0]
    case = sample_id.rsplit("_", 1)[0]
    return "WMH:" + case, case
