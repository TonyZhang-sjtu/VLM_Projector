from src.preprocessing.neurovfm import NeuroVFMPreprocessor

PREPROCESSORS = {'neurovfm':NeuroVFMPreprocessor}


def register_preprocessor(name, factory):
    if name in PREPROCESSORS:
        raise ValueError(f'Preprocessor already registered: {name}')
    PREPROCESSORS[name] = factory


def build_preprocessor(config):
    return PREPROCESSORS[config.preprocessing.name](config)
