from src.preprocessing.neurovfm import NeuroVFMPreprocessor

PREPROCESSORS = {'neurovfm':NeuroVFMPreprocessor}


def register_preprocessor(name, factory):
    if name in PREPROCESSORS:
        raise ValueError(f'Preprocessor already registered: {name}')
    PREPROCESSORS[name] = factory


def build_preprocessor(config):
    if config.preprocessing.name == 'neuroqa_mni1mm_v2':
        from src.preprocessing.neuroqa_v2 import NeuroQAV2Preprocessor
        return NeuroQAV2Preprocessor(config)
    if config.preprocessing.name == 'neuroqa_mni1mm':
        from src.preprocessing.neuroqa import NeuroQAPreprocessor
        return NeuroQAPreprocessor(config)
    return PREPROCESSORS[config.preprocessing.name](config)
