from src.models.encoders.neurovfm import NeuroVFMEncoder
ENCODERS = {'neurovfm':NeuroVFMEncoder}


def register_encoder(name, factory):
    if name in ENCODERS:
        raise ValueError(f'Encoder already registered: {name}')
    ENCODERS[name] = factory


def build_encoder(config):
    return ENCODERS[config.encoder.name](config)
