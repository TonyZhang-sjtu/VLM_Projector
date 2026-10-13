import gc
import pytest
import torch
from src.config import load_config, ROOT
from src.data.manifest import load_manifests
from src.preprocessing.registry import build_preprocessor
from src.models.encoders.registry import build_encoder
from src.models.projectors.registry import build_projector

DEVICE = 'cuda:0'


@pytest.mark.parametrize('stage', ['a','b_lora','b_full'])
def test_stage_gradients(stage):
    c = load_config([ROOT/f'configs/training/stage_{stage}.yaml'] if stage == 'a' else
        [ROOT/f'configs/training/stage_{stage}.yaml'], overrides=['training.feature_cache=false'])
    sample = load_manifests(c)[0]['train'][0]
    volume = build_preprocessor(c)(sample)
    encoder = build_encoder(c).to(DEVICE)
    projector = build_projector(c,encoder.feature_dim,4096).to(DEVICE)
    assert sum(p.numel() for p in projector.parameters()) == 19931136
    output = encoder([volume])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        projected = projector(output.features[0],output.coords[0],output.metadata[0])
        loss = projected.embeddings.float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters())
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == {'a':0,'b_lora':442368,'b_full':85803234}[stage]
    assert all(p.grad is None for p in encoder.parameters() if not p.requires_grad)
    if stage != 'a':
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        assert any(torch.count_nonzero(p.grad) for p in trainable)
    del encoder,projector,output,projected,loss,trainable
    gc.collect(); torch.cuda.empty_cache()


def test_independent_batch_isolation_and_cache(tmp_path):
    c = load_config(overrides=['training.feature_cache=false',f'training.feature_cache_dir={tmp_path}'])
    rows = load_manifests(c, datasets=['all'])[0]['train']
    samples = [rows[0],next(s for s in rows if s.cohort == 'ISLES22')]
    pre = build_preprocessor(c)
    volumes = [pre(s) for s in samples]
    encoder = build_encoder(c).to(DEVICE).eval()
    packed = encoder(volumes).features
    solo = [encoder([v]).features[0] for v in volumes]
    for a,b in zip(packed,solo):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    c.training.feature_cache = True
    cached = encoder(volumes).features
    again = encoder(volumes).features
    for a,b,d in zip(solo,cached,again):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(b,d,rtol=0,atol=0)


def test_deterministic_attention_preserves_forward():
    from src.models.encoders.deterministic import enable_deterministic_attention
    c=load_config(overrides=['training.feature_cache=false'])
    sample=next(s for s in load_manifests(c,datasets=['all'])[0]['train'] if s.cohort=='ISLES22')
    volume=build_preprocessor(c)(sample)
    encoder=build_encoder(c).to(DEVICE).eval()
    before=encoder([volume]).features[0]
    enable_deterministic_attention(encoder.core)
    after=encoder([volume]).features[0]
    torch.testing.assert_close(before,after,rtol=0,atol=0)
