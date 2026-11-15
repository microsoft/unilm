"""CPU tests: python -m pytest test_long_documents.py"""

import copy

import pytest
import torch
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification

from long_documents import extend_position_embeddings, use_lut_attention


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = LayoutLMv3Config(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                              intermediate_size=64, max_position_embeddings=66, input_size=64, patch_size=16,
                              max_2d_position_embeddings=1024, coordinate_size=6, shape_size=4, num_labels=5,
                              rel_pos_bins=32, max_rel_pos=128, max_rel_2d_pos=256, rel_2d_pos_bins=64)
    return LayoutLMv3ForTokenClassification(config).eval()


def batch(b=2, t=48, pad=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randint(0, 900, (b, t, 1), generator=g)
    y0 = torch.randint(0, 900, (b, t, 1), generator=g)
    mask = torch.ones(b, t, dtype=torch.long)
    mask[1, t - pad:] = 0
    return dict(input_ids=torch.randint(5, 100, (b, t), generator=g), bbox=torch.cat([x0, y0, x0 + 60, y0 + 20], -1),
                attention_mask=mask, pixel_values=torch.randn(b, 3, 64, 64, generator=g))


@pytest.mark.parametrize("pad", [0, 10])
def test_lut_attention_matches_eager(pad):
    eager = tiny_model()
    lut = use_lut_attention(copy.deepcopy(eager))
    inputs = batch(pad=pad)
    with torch.no_grad():
        expected, got = eager(**inputs).logits, lut(**inputs).logits
    keep = inputs["attention_mask"].bool()
    torch.testing.assert_close(got[keep], expected[keep], rtol=1e-4, atol=1e-5)


def test_extending_positions_keeps_short_inputs_unchanged():
    model = tiny_model()
    inputs = batch(t=40)
    with torch.no_grad():
        before = model(**inputs).logits
    extend_position_embeddings(model, 256)
    emb = model.layoutlmv3.embeddings.position_embeddings
    assert emb.num_embeddings == 256 + emb.padding_idx + 1 == model.config.max_position_embeddings
    learned = emb.weight[emb.padding_idx + 1:]
    torch.testing.assert_close(learned[64:128], learned[:64], rtol=0, atol=0)   # tiled
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs).logits, before, rtol=0, atol=0)


def test_long_input_trains_with_both_changes():
    model = use_lut_attention(extend_position_embeddings(tiny_model(), 256)).train()
    inputs = batch(t=200)
    labels = torch.randint(0, 5, inputs["input_ids"].shape)
    loss = model(**inputs, labels=labels).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.layoutlmv3.embeddings.position_embeddings.weight.grad[150].abs().sum() > 0
