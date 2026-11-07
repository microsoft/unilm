import random

import numpy as np
import pytest
import torch
from transformers import LayoutLMv3Config

from layoutlmv3_pretraining import (KMeansPatchTokenizer, LayoutLMv3ForPreTraining, bbox_to_patches,
                                    blockwise_mask, span_mask, wpa_labels)


def tiny_config():
    return LayoutLMv3Config(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=64, max_position_embeddings=80, input_size=224, patch_size=16,
                            max_2d_position_embeddings=1024, coordinate_size=6, shape_size=4,
                            rel_pos_bins=32, max_rel_pos=128, max_rel_2d_pos=256, rel_2d_pos_bins=64)


def tiny_batch(b=2, t=16, vocab=100, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, vocab, (b, t), generator=g)
    x0 = torch.randint(0, 900, (b, t, 1), generator=g); y0 = torch.randint(0, 900, (b, t, 1), generator=g)
    bbox = torch.cat([x0, y0, x0 + 50, y0 + 20], -1)
    mask_pos = torch.zeros(b, 196, dtype=torch.bool); mask_pos[:, :78] = True
    return dict(input_ids=ids, bbox=bbox, attention_mask=torch.ones(b, t, dtype=torch.long),
                pixel_values=torch.randn(b, 3, 224, 224, generator=g), bool_masked_pos=mask_pos,
                mlm_labels=torch.where(torch.rand(b, t, generator=g) < 0.3, ids, -100),
                mim_labels=torch.where(mask_pos, torch.randint(0, k, (b, 196), generator=g), -100),
                wpa_labels=torch.randint(0, 2, (b, t), generator=g))


# ------------------------------------------------------------------ masking

def test_span_mask_ratio_and_contiguity():
    rng = np.random.default_rng(0)
    cand = list(range(1, 201))
    m = span_mask(cand, 0.3, 3.0, rng)
    assert len(m) == 60 and set(m) <= set(cand)
    runs = sum(1 for i, p in enumerate(m) if i == 0 or m[i - 1] != p - 1)
    assert runs < len(m) / 1.5  # spans, not isolated tokens


def test_blockwise_mask_exact_count():
    for seed in range(20):
        m = blockwise_mask(14, 14, 0.4, rng=random.Random(seed))
        assert m.shape == (14, 14) and m.sum() == round(196 * 0.4)


def test_bbox_to_patches():
    assert bbox_to_patches([0, 0, 1000, 1000]) == list(range(196))
    assert bbox_to_patches([0, 0, 71, 71]) == [0]           # inside the first 1000/14 cell
    assert bbox_to_patches([0, 0, 72, 10]) == [0, 1]        # crosses into column 1
    assert bbox_to_patches([0, 0, 0, 0]) == []              # special tokens


def test_wpa_labels():
    bbox = torch.tensor([[0, 0, 0, 0], [0, 0, 50, 50], [500, 500, 550, 550], [0, 0, 50, 50]])
    special = torch.tensor([True, False, False, False])
    text_mask = torch.tensor([False, False, False, True])
    img = torch.zeros(14, 14, dtype=torch.bool); img[7, 7] = True  # patch covering (500..571)
    assert wpa_labels(bbox, text_mask, img, special).tolist() == [-100, 1, 0, -100]


# ------------------------------------------------------------------ model

def test_all_three_losses_backpropagate_into_backbone():
    """Regression guard: every objective must produce a gradient on the shared encoder."""
    torch.manual_seed(0)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    batch = tiny_batch()
    probe = model.layoutlmv3.encoder.layer[0].attention.self.query.weight
    for name in ("mlm_loss", "mim_loss", "wpa_loss"):
        model.zero_grad()
        out = model(**batch)
        loss = getattr(out, name)
        assert loss.requires_grad and torch.isfinite(loss)
        loss.backward()
        assert probe.grad is not None and probe.grad.abs().sum() > 0, name
    model.zero_grad()
    out = model(**batch)
    torch.testing.assert_close(out.loss, out.mlm_loss + out.mim_loss + out.wpa_loss)
    out.loss.backward()
    for head in (model.mlm_head.dense, model.mim_head[1], model.wpa_head.net[0], model.wpa_head.net[2]):
        assert head.weight.grad.abs().sum() > 0
    assert model.layoutlmv3.image_mask_token.grad.abs().sum() > 0


def test_image_mask_replaces_patches_only_where_masked():
    torch.manual_seed(0)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8).eval()
    b = tiny_batch(b=1)
    with torch.no_grad():
        clean = model.layoutlmv3.forward_image(b["pixel_values"])
        model.layoutlmv3._bool_masked_pos = b["bool_masked_pos"]
        masked = model.layoutlmv3.forward_image(b["pixel_values"])
        model.layoutlmv3._bool_masked_pos = None
    diff = (clean - masked).abs().sum(-1)[0, 1:]               # skip [CLS]
    assert (diff[b["bool_masked_pos"][0]] > 0).all() and (diff[~b["bool_masked_pos"][0]] == 0).all()


def test_mlm_decoder_tied_to_word_embeddings():
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    assert model.mlm_head.decoder.weight is model.layoutlmv3.embeddings.word_embeddings.weight


def test_kmeans_tokenizer_roundtrip(tmp_path):
    pv = [torch.randn(2, 3, 224, 224) for _ in range(2)]
    tok = KMeansPatchTokenizer.fit(pv, vocab_size=16, seed=0)
    ids = tok.encode(pv[0])
    assert ids.shape == (2, 196) and ids.max() < 16
    tok.save(tmp_path / "t.npz")
    assert torch.equal(KMeansPatchTokenizer.load(tmp_path / "t.npz").encode(pv[0]), ids)


# ------------------------------------------------------------------ DALL-E dVAE tokenizer

def _dalle_weights():
    import os
    from pathlib import Path
    for p in (os.environ.get("DALLE_ENCODER"), "runs/dalle/encoder.pkl",
              Path.home() / ".cache" / "layoutlmv3_pretraining" / "dalle_encoder.pkl"):
        if p and Path(p).exists():
            return str(p)
    return None


def test_dalle_tokenizer_preprocessing_inverts_processor_normalisation():
    from layoutlmv3_pretraining.dvae import DalleEncoder, DalleTokenizer, LOGIT_LAPLACE_EPS
    tok = DalleTokenizer(DalleEncoder(n_hid=8, vocab_size=16))
    white = torch.ones(1, 3, 224, 224)                    # processor output for a white page: (1 - 0.5) / 0.5
    black = -torch.ones(1, 3, 224, 224)
    x_w, x_b = tok.preprocess(white), tok.preprocess(black)
    assert x_w.shape == (1, 3, 112, 112)
    torch.testing.assert_close(x_w, torch.full_like(x_w, 1 - LOGIT_LAPLACE_EPS))
    torch.testing.assert_close(x_b, torch.full_like(x_b, LOGIT_LAPLACE_EPS))
    ids = tok.encode(torch.rand(2, 3, 224, 224) * 2 - 1)
    assert ids.shape == (2, 196) and ids.dtype == torch.long and int(ids.max()) < 16 and tok.vocab_size == 16


@pytest.mark.skipif(_dalle_weights() is None, reason="DALL-E encoder.pkl not available (set DALLE_ENCODER)")
def test_dalle_tokenizer_with_openai_weights():
    from layoutlmv3_pretraining import DalleTokenizer
    tok = DalleTokenizer.from_pretrained(_dalle_weights())
    assert tok.vocab_size == 8192
    g = torch.Generator().manual_seed(0)
    pv = torch.rand(2, 3, 224, 224, generator=g) * 2 - 1
    ids = tok.encode(pv)
    assert ids.shape == (2, 196) and ids.dtype == torch.long and 0 <= int(ids.min()) and int(ids.max()) < 8192
    # a blank page is dominated by one token (borders differ because of zero padding); noise is not
    blank = tok.encode(torch.ones(1, 3, 224, 224))
    assert torch.bincount(blank[0]).max() > 196 // 2 and ids[0].unique().numel() > 20


def test_collator_without_image_tokenizer_defers_mim_labels():
    from layoutlmv3_pretraining import KMeansPatchTokenizer
    from layoutlmv3_pretraining.data import PretrainingCollator, add_mim_labels

    class Tok:
        all_special_ids, mask_token_id = [0, 1, 2], 3

    b = tiny_batch(b=2, t=16)
    feats = [{k: b[k][i] for k in ("input_ids", "bbox", "attention_mask", "pixel_values")} for i in range(2)]
    out = PretrainingCollator(Tok(), None, seed=0)(feats)
    assert "mim_labels" not in out and out["bool_masked_pos"].sum(1).tolist() == [78, 78]
    tok = KMeansPatchTokenizer(np.random.default_rng(0).normal(size=(8, 16)))
    add_mim_labels(out, tok)
    m = out["bool_masked_pos"]
    assert (out["mim_labels"][~m] == -100).all() and (out["mim_labels"][m] >= 0).all()


# ------------------------------------------------------------------ IDL corpus

def test_line_boxes_to_words_normalises_to_0_1000():
    from layoutlmv3_pretraining.corpus import line_boxes_to_words
    words, boxes = line_boxes_to_words(["Hello world", "low", "x"], [[0.1, 0.2, 0.3, 0.05], [0, 0, 1, 1],
                                                                     [0.95, 0.9, 0.2, 0.2]],
                                       scores=[0.99, 0.1, 0.9], min_score=0.5)
    assert words == ["Hello", "world", "x"]
    assert boxes == [[100, 200, 400, 250], [100, 200, 400, 250], [950, 900, 1000, 1000]]


def test_idl_pages_from_webdataset_tar(tmp_path):
    import io, json, tarfile
    from PIL import Image
    from layoutlmv3_pretraining.corpus import IDLPages, NgramBlocklist
    page = {"text": ["one two three four five six seven", "eight nine ten"],
            "bbox": [[0.1, 0.1, 0.5, 0.02], [0.1, 0.2, 0.3, 0.02]], "score": [0.9, 0.9]}
    frames = [Image.new("1", (1000, 1300), 1) for _ in range(3)]
    tif = io.BytesIO(); frames[0].save(tif, format="TIFF", save_all=True, append_images=frames[1:])
    members = {"doc1.json": json.dumps({"pages": [page] * 3}).encode(), "doc1.tif": tif.getvalue(),
               "doc2.json": json.dumps({"pages": [{"text": ["too short"], "bbox": [[0, 0, 1, 1]]}]}).encode(),
               "doc2.tif": tif.getvalue()}
    path = tmp_path / "shard.tar"
    with tarfile.open(path, "w") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name); info.size = len(data); t.addfile(info, io.BytesIO(data))

    class Proc:  # stands in for LayoutLMv3Processor
        def __call__(self, images, words, boxes, **kw):
            n = len(words[0])
            return {"input_ids": torch.ones(1, n, dtype=torch.long), "bbox": torch.tensor([boxes[0]]),
                    "attention_mask": torch.ones(1, n, dtype=torch.long),
                    "pixel_values": torch.zeros(1, 3, *images[0].size[::-1])}

    ds = IDLPages([str(path)], Proc(), max_pages_per_doc=2)
    pages = list(ds.raw_pages())
    assert [p["key"] for p in pages] == ["doc1/0", "doc1/1"]           # 2 pages max, doc2 too short
    assert pages[0]["image"].mode == "L" and len(pages[0]["tokens"]) == 10 and pages[0]["rotation"] == 0
    assert pages[0]["bboxes"][0] == [100, 100, 600, 120]
    items = list(ds)
    assert len(items) == 2 and items[0]["bbox"].shape == (10, 4)
    block = NgramBlocklist([page["text"][0].split()], n=5)
    assert block(pages[0]["tokens"]) and not block(["completely", "different", "words", "on", "this", "page"])
    assert list(IDLPages([str(path)], Proc(), exclude=block).raw_pages()) == []


@pytest.mark.parametrize("hidden,layers,heads,coord,shape", [(32, 2, 2, 6, 4), (48, 3, 4, 8, 8)])
def test_from_backbone_sizes_heads_from_config(tmp_path, hidden, layers, heads, coord, shape):
    """Heads follow the backbone's hidden size / depth (base: 768 x 12, large: 1024 x 24)."""
    from transformers import LayoutLMv3Model
    cfg = tiny_config()
    cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.intermediate_size = \
        hidden, layers, heads, 2 * hidden
    cfg.coordinate_size, cfg.shape_size = coord, shape          # 4 * coord + 2 * shape == hidden
    LayoutLMv3Model(cfg).save_pretrained(tmp_path)
    model = LayoutLMv3ForPreTraining.from_backbone(str(tmp_path), image_vocab_size=8192)
    assert model.mlm_head.dense.weight.shape == (hidden, hidden)
    assert model.mlm_head.decoder.weight is model.layoutlmv3.embeddings.word_embeddings.weight
    assert model.mim_head[1].weight.shape == (8192, hidden) and model.config.wpa_head == "mlp"
    assert model.wpa_head.net[0].weight.shape == (hidden, hidden) and model.wpa_head.net[2].weight.shape == (2, hidden)
    assert len(model.layoutlmv3.encoder.layer) == layers
    out = model(**tiny_batch(k=8192))
    assert out.mim_logits.shape == (2, 196, 8192) and torch.isfinite(out.loss)


# ------------------------------------------------------------------ head warm-up

def test_new_parameters_are_heads_and_image_mask_token_only():
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    names = set(model.new_parameters())
    assert "layoutlmv3.image_mask_token" in names
    assert all(n.startswith(("mlm_head.", "mim_head.", "wpa_head.")) for n in names - {"layoutlmv3.image_mask_token"})
    assert "mlm_head.decoder.bias" in names
    # the tied decoder weight is the word embedding matrix, i.e. an encoder parameter
    assert not any(p is model.layoutlmv3.embeddings.word_embeddings.weight for p in model.new_parameters().values())


def test_head_warmup_freezes_encoder_and_trains_heads():
    """With the encoder frozen, an optimiser step changes the new parameters and nothing else."""
    torch.manual_seed(0)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    model.set_encoder_trainable(False)
    new = model.new_parameters()
    assert all(p.requires_grad == (n in new) for n, p in model.named_parameters())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(new.values(), lr=1e-2, weight_decay=0.01)
    model(**tiny_batch()).loss.backward()
    assert all(p.grad is None for n, p in model.named_parameters() if n not in new)
    opt.step()
    for n, p in model.named_parameters():
        changed = not torch.equal(p, before[n])
        assert changed == (n in new), n
    model.set_encoder_trainable(True)
    assert all(p.requires_grad for p in model.parameters())


def test_layerwise_param_groups():
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)   # 2 layers -> depths 0..3
    groups = model.layerwise_param_groups(1e-4, layer_decay=0.5, new_lr=1e-3)
    by_name = {n: g["lr"] for g in groups for n in g["names"]}
    assert len(by_name) == sum(1 for _ in model.parameters())             # every parameter exactly once
    assert by_name["layoutlmv3.embeddings.word_embeddings.weight"] == pytest.approx(1e-4 * 0.5 ** 3)
    assert by_name["layoutlmv3.patch_embed.proj.weight"] == pytest.approx(1e-4 * 0.5 ** 3)
    assert by_name["layoutlmv3.encoder.layer.0.attention.self.query.weight"] == pytest.approx(1e-4 * 0.5 ** 2)
    assert by_name["layoutlmv3.encoder.layer.1.output.dense.weight"] == pytest.approx(1e-4 * 0.5)
    assert by_name["layoutlmv3.encoder.rel_pos_bias.weight"] == pytest.approx(1e-4)
    assert by_name["mim_head.1.weight"] == by_name["layoutlmv3.image_mask_token"] == pytest.approx(1e-3)
    no_override = {n: g["lr"] for g in model.layerwise_param_groups(1e-4) for n in g["names"]}
    assert set(no_override.values()) == {1e-4}


def test_from_backbone_random_init_does_not_load_weights(tmp_path):
    from transformers import LayoutLMv3Model
    torch.manual_seed(0)
    LayoutLMv3Model(tiny_config()).save_pretrained(tmp_path)
    loaded = LayoutLMv3ForPreTraining.from_backbone(str(tmp_path), image_vocab_size=8)
    assert loaded.loading_info["encoder_keys_not_in_checkpoint"] == ["image_mask_token"]
    torch.manual_seed(1)
    scratch = LayoutLMv3ForPreTraining.from_backbone(str(tmp_path), image_vocab_size=8, random_init=True)
    w = "encoder.layer.0.attention.self.query.weight"
    assert not torch.equal(loaded.layoutlmv3.state_dict()[w], scratch.layoutlmv3.state_dict()[w])
    assert scratch.mlm_head.decoder.weight is scratch.layoutlmv3.embeddings.word_embeddings.weight


# ------------------------------------------------------------------ WPA head variants

def test_token_patch_matrix_matches_bbox_to_patches():
    from layoutlmv3_pretraining.modeling import token_patch_matrix
    g = torch.Generator().manual_seed(0)
    x0 = torch.randint(0, 1000, (3, 40), generator=g); y0 = torch.randint(0, 1000, (3, 40), generator=g)
    x1 = (x0 + torch.randint(-50, 300, (3, 40), generator=g)).clamp(max=1000)
    y1 = (y0 + torch.randint(-50, 300, (3, 40), generator=g)).clamp(max=1000)
    bbox = torch.stack([x0, y0, x1, y1], -1)
    bbox[0, 0] = torch.tensor([0, 0, 0, 0])                                   # e.g. a special token
    m = token_patch_matrix(bbox)
    for b in range(3):
        for t in range(40):
            assert torch.nonzero(m[b, t]).flatten().tolist() == bbox_to_patches(bbox[b, t].tolist()), (b, t)


@pytest.mark.parametrize("variant", ["linear", "mlp", "mlp_patch"])
def test_wpa_head_variants(variant):
    torch.manual_seed(0)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8, wpa_head=variant)
    assert model.config.wpa_head == variant
    batch = tiny_batch()
    out = model(**batch)
    assert out.wpa_logits.shape == (2, 16, 2)
    out.wpa_loss.backward()
    assert all(p.grad is not None for p in model.wpa_head.parameters())
    assert model.layoutlmv3.encoder.layer[0].attention.self.query.weight.grad.abs().sum() > 0
    assert set(model.new_parameters()) >= {f"wpa_head.{n}" for n, _ in model.wpa_head.named_parameters()}


def test_wpa_head_default_is_the_papers_two_layer_mlp():
    """The paper's WPA head is a two-layer MLP; mlp (text token only) is the default everywhere, never mlp_patch."""
    import importlib.util
    import pathlib
    from layoutlmv3_pretraining.modeling import DEFAULT_WPA_HEAD, WPAHead
    assert DEFAULT_WPA_HEAD == "mlp"
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    assert model.config.wpa_head == "mlp" and model.wpa_head.variant == "mlp"
    assert [type(m).__name__ for m in model.wpa_head.net] == ["Linear", "GELU", "Linear"]
    assert model.wpa_head.net[0].in_features == tiny_config().hidden_size          # text token only: no patches
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "pretrain.py"
    spec = importlib.util.spec_from_file_location("pretrain_script", path)
    pretrain = importlib.util.module_from_spec(spec); spec.loader.exec_module(pretrain)
    assert pretrain.build_parser().parse_args(["--output_dir", "x"]).wpa_head == "mlp"
    # saved heads identify their variant, including heads saved before WPAHead existed (linear runs)
    for variant in WPAHead.VARIANTS:
        heads = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8, wpa_head=variant).state_dict()
        assert WPAHead.variant_of(heads) == variant
    assert WPAHead.variant_of({"wpa_head.weight": torch.zeros(2, 32), "wpa_head.bias": torch.zeros(2)}) == "linear"


def test_save_train_state_roundtrip_is_atomic(tmp_path):
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "pretrain.py"
    spec = importlib.util.spec_from_file_location("pretrain_script", path)
    pretrain = importlib.util.module_from_spec(spec); spec.loader.exec_module(pretrain)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(**tiny_batch()).loss.backward(); opt.step()
    target = tmp_path / "train_state.pt"
    pretrain.save_train_state(target, model=model.state_dict(), optimizer=opt.state_dict(), step=7, log=[{"step": 0}])
    assert sorted(x.name for x in tmp_path.iterdir()) == ["train_state.pt"]           # no temporary file left
    state = torch.load(target, weights_only=False)
    fresh = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    fresh.load_state_dict(state["model"])
    for (n, p), q in zip(model.state_dict().items(), fresh.state_dict().values()):
        torch.testing.assert_close(p, q, msg=n)
    opt2 = torch.optim.AdamW(fresh.parameters(), lr=1e-3); opt2.load_state_dict(state["optimizer"])
    assert state["step"] == 7 and opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_wpa_mlp_patch_reads_the_covered_patches():
    """mlp_patch: a token's logits depend on the hidden states of the patches its box covers, not on others."""
    from layoutlmv3_pretraining.modeling import WPAHead
    torch.manual_seed(0)
    head = WPAHead(tiny_config(), "mlp_patch")
    text, patches = torch.randn(1, 1, 32), torch.randn(1, 196, 32)
    bbox = torch.tensor([[[0, 0, 70, 70]]])                                   # the top-left patch only
    covered = bbox_to_patches(bbox[0, 0].tolist())
    base = head(text, patches, bbox)
    other = patches.clone(); other[0, [p for p in range(196) if p not in covered]] += 5.0
    torch.testing.assert_close(head(text, other, bbox), base)
    touched = patches.clone(); touched[0, covered] += 5.0
    assert not torch.allclose(head(text, touched, bbox), base)


def test_from_backbone_round_trip_keeps_image_mask_token(tmp_path):
    """A backbone saved by pretrain.py (LayoutLMv3ModelWithImageMask) reloads with its trained [MASK] token."""
    torch.manual_seed(0)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    with torch.no_grad():
        model.layoutlmv3.image_mask_token.fill_(0.123)
    model.layoutlmv3.save_pretrained(tmp_path)
    again = LayoutLMv3ForPreTraining.from_backbone(str(tmp_path), image_vocab_size=8)
    torch.testing.assert_close(again.layoutlmv3.image_mask_token, model.layoutlmv3.image_mask_token)
    assert again.loading_info["image_mask_token_from_checkpoint"] is True
    w = "encoder.layer.1.output.dense.weight"
    torch.testing.assert_close(again.layoutlmv3.state_dict()[w], model.layoutlmv3.state_dict()[w])


# ------------------------------------------------------------------ screen options

def test_freeze_bottom_layers():
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)        # 2 layers
    model.set_encoder_trainable(True)
    model.freeze_bottom(1)                                                     # embeddings + layer 0
    for name, p in model.named_parameters():
        frozen = name in ("layoutlmv3.embeddings.word_embeddings.weight", "layoutlmv3.patch_embed.proj.weight") \
            or name.startswith("layoutlmv3.encoder.layer.0.")
        if frozen:
            assert not p.requires_grad, name
        if name.startswith(("layoutlmv3.encoder.layer.1.", "mlm_head.", "mim_head.", "wpa_head.",
                            "layoutlmv3.encoder.rel_pos")) or name == "layoutlmv3.image_mask_token":
            assert p.requires_grad, name
    assert not model.mlm_head.decoder.weight.requires_grad                    # tied to the word embeddings


def test_encoder_drift():
    torch.manual_seed(0)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    ref = model.checkpoint_state()
    assert "layoutlmv3.image_mask_token" not in ref and not any(k.startswith("mim_head") for k in ref)
    assert model.encoder_drift(ref)["all"] == 0.0
    with torch.no_grad():
        w = model.layoutlmv3.encoder.layer[1].output.dense.weight
        w.mul_(1.1)                                                            # +10% on one tensor of layer 1
        model.mim_head[1].weight.add_(1.0)                                     # heads are not counted
    d = model.encoder_drift(ref)
    assert d["depth_1"] == 0.0 and d["depth_0"] == 0.0 and 0 < d["depth_2"] < 0.1 and 0 < d["all"] < d["depth_2"]


def test_idl_pages_skips_malformed_tiff(monkeypatch):
    # IDL contains TIFFs on which PIL raises SyntaxError ("unknown data organization") when counting
    # or seeking frames; such a document must be skipped, not stop the DataLoader worker.
    import io, json
    from PIL import Image
    from layoutlmv3_pretraining import corpus
    page = {"text": ["one two three four five six seven", "eight nine ten"],
            "bbox": [[0.1, 0.1, 0.5, 0.02], [0.1, 0.2, 0.3, 0.02]], "score": [0.9, 0.9]}
    good = io.BytesIO(); Image.new("1", (100, 130), 1).save(good, format="TIFF")
    sample = {"__key__": "d", "json": json.dumps({"pages": [page]}).encode(), "tif": good.getvalue()}
    assert len(list(corpus.idl_pages(sample))) == 1
    assert list(corpus.idl_pages({**sample, "tif": b"II*\x00not a tiff"})) == []

    class Broken:  # what PIL returns for the malformed files: opening works, frame access raises
        @property
        def n_frames(self):
            raise SyntaxError("unknown data organization")

    monkeypatch.setattr(corpus.Image, "open", lambda f: Broken())
    assert list(corpus.idl_pages(sample)) == []

    class BrokenSeek(Broken):
        n_frames = 1

        def seek(self, i):
            raise SyntaxError("unknown data organization")

    monkeypatch.setattr(corpus.Image, "open", lambda f: BrokenSeek())
    assert list(corpus.idl_pages(sample)) == []


# ------------------------------------------------------------------ data-pipeline fixes (scale-v2)

def _text_page(lines, size=(1000, 1300)):
    """An upright page with 'text' (vertical strokes) inside each 0-1000 line box, and the OCR json page."""
    from PIL import Image, ImageDraw
    w, h = size
    im = Image.new("L", size, 255)
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in lines:
        for x in range(x0 * w // 1000, x1 * w // 1000, 6):
            d.rectangle([x, y0 * h // 1000, x + 2, y1 * h // 1000], fill=0)
    page = {"text": [" ".join(f"w{i}{j}" for j in range(6)) for i in range(len(lines))],
            "bbox": [[x0 / 1000, y0 / 1000, (x1 - x0) / 1000, (y1 - y0) / 1000] for x0, y0, x1, y1 in lines],
            "score": [0.99] * len(lines)}
    return im, page


LINES = [(80, 60 + 40 * i, 80 + 60 * (i % 7) + 300, 75 + 40 * i) for i in range(14)]   # upper part, ragged right


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_page_orientation_finds_the_turn_that_fits_the_boxes(k):
    from PIL import Image
    from layoutlmv3_pretraining.corpus import line_boxes_to_words, page_orientation
    upright, page = _text_page(LINES)
    _, boxes = line_boxes_to_words(page["text"], page["bbox"])
    # stored frame = upright turned clockwise by k quarter turns; k counter-clockwise turns undo it
    stored = upright.rotate(-90 * k, expand=True)
    assert page_orientation(stored, boxes) == k
    assert page_orientation(stored.transpose(Image.Transpose.ROTATE_90).rotate(-90, expand=True), boxes) == k
    assert page_orientation(Image.new("L", (300, 400), 255), boxes) == 0          # blank page: keep as stored


def test_idl_pages_turns_or_drops_rotated_frames():
    import io, json
    from layoutlmv3_pretraining.corpus import idl_pages, page_orientation
    upright, page = _text_page(LINES)
    frames = [upright, upright.rotate(-90, expand=True), upright.rotate(-270, expand=True),
              upright.rotate(180, expand=True)]                     # as stored: upright, 90 CW, 90 CCW, upside down
    tif = io.BytesIO(); frames[0].save(tif, format="TIFF", save_all=True, append_images=frames[1:])
    sample = {"__key__": "d", "json": json.dumps({"pages": [page] * 4}).encode(), "tif": tif.getvalue()}

    stats = {}
    fixed = list(idl_pages(sample, max_pages_per_doc=4, stats=stats))
    assert [(p["key"], p["rotation"]) for p in fixed] == [("d/0", 0), ("d/1", 90), ("d/2", 270)]   # 180 dropped
    assert stats == {"turned_90": 1, "turned_270": 1, "dropped_rotated_180": 1}
    for p in fixed:                                                  # every kept page now fits its boxes as is
        assert p["image"].size[0] < p["image"].size[1] and page_orientation(p["image"], p["bboxes"]) == 0
    assert [p["key"] for p in idl_pages(sample, max_pages_per_doc=4, orientation="drop")] == ["d/0"]
    off = list(idl_pages(sample, max_pages_per_doc=4, orientation="off"))
    assert [p["rotation"] for p in off] == [0, 0, 0, 0] and off[1]["image"].size[0] > off[1]["image"].size[1]
    with pytest.raises(ValueError):
        list(idl_pages(sample, orientation="sideways"))


class _ConstantPages(torch.utils.data.IterableDataset):
    """Identical features from every worker: any difference between masks comes from the collator's RNG."""

    def __iter__(self):
        ids = torch.arange(5, 69); ids[0], ids[-1] = 0, 2
        for _ in range(4):
            yield {"input_ids": ids, "bbox": torch.tensor([[0, 0, 500, 500]] * 64),
                   "attention_mask": torch.ones(64, dtype=torch.long), "pixel_values": torch.zeros(3, 4, 4)}


class _Tok:
    all_special_ids, mask_token_id = [0, 1, 2], 3


def _masks(seed, workers):
    from torch.utils.data import DataLoader
    from layoutlmv3_pretraining.data import PretrainingCollator
    dl = DataLoader(_ConstantPages(), batch_size=2, num_workers=workers,
                    collate_fn=PretrainingCollator(_Tok(), None, seed=seed))
    return [(b["mlm_labels"] != -100, b["bool_masked_pos"]) for b in dl]


def test_collator_draws_different_masks_in_each_worker():
    batches = _masks(42, workers=2)                 # in-order loading: batch 2i from worker 0, 2i + 1 from worker 1
    assert len(batches) == 4
    for a, b in ((batches[0], batches[1]), (batches[2], batches[3])):
        assert not torch.equal(a[0], b[0]) and not torch.equal(a[1], b[1])
    again = _masks(42, workers=2)                   # reproducible run to run
    assert all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(batches, again))
    other = _masks((42, 2000), workers=2)           # a resumed run (seed, step) draws a new stream
    assert not torch.equal(batches[0][0], other[0][0])


def test_validation_masks_fixed_in_the_main_process():
    from layoutlmv3_pretraining.data import PretrainingCollator
    feats = list(_ConstantPages())
    a = PretrainingCollator(_Tok(), None, seed=0)(feats)
    b = PretrainingCollator(_Tok(), None, seed=0)(feats)
    assert torch.equal(a["mlm_labels"], b["mlm_labels"]) and torch.equal(a["bool_masked_pos"], b["bool_masked_pos"])
    # an int seed in the main process is the same stream as before the per-worker seeding (old validation masks)
    expected = span_mask(list(range(1, 63)), 0.3, 3.0, np.random.default_rng(0))
    assert (a["mlm_labels"][0] != -100).nonzero().flatten().tolist() == expected
    assert torch.equal(a["bool_masked_pos"][0], torch.from_numpy(blockwise_mask(14, 14, 0.4, rng=random.Random(0))).reshape(-1))


class _KeyProcessor:
    """Stands in for LayoutLMv3Processor; the key travels in the item."""

    def __call__(self, images, words, boxes, **kw):
        return {k: torch.zeros(1, 1, dtype=torch.long) for k in ("input_ids", "bbox", "attention_mask", "pixel_values")}


def _keys(batch):
    return [x["key"] for x in batch]


def _idl_tars(tmp_path, n_tars=3, docs=15):
    import io, json, tarfile
    from PIL import Image
    page = {"text": ["one two three four five six seven", "eight nine ten"],
            "bbox": [[0.1, 0.1, 0.5, 0.02], [0.1, 0.2, 0.3, 0.02]], "score": [0.9, 0.9]}
    tif = io.BytesIO(); Image.new("L", (100, 130), 255).save(tif, format="TIFF")
    paths = []
    for t in range(n_tars):
        path = tmp_path / f"shard{t}.tar"
        with tarfile.open(path, "w") as tar:
            for d in range(docs):
                for name, data in ((f"t{t}d{d:02d}.json", json.dumps({"pages": [page] * 2}).encode()),
                                   (f"t{t}d{d:02d}.tif", tif.getvalue())):
                    info = tarfile.TarInfo(name); info.size = len(data); tar.addfile(info, io.BytesIO(data))
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("workers", [0, 2, 3])
def test_resumed_stream_continues_without_replay(tmp_path, workers):
    import itertools
    from torch.utils.data import DataLoader
    from layoutlmv3_pretraining.corpus import IDLPages
    sources = _idl_tars(tmp_path)                       # 45 pages (single-frame TIFFs: one page per document)

    def stream(skip=0):
        ds = IDLPages(sources, _KeyProcessor(), shuffle_buffer=7, repeat=True, seed=5).skip_batches(skip, 4)
        dl = DataLoader(ds, batch_size=4, collate_fn=_keys, num_workers=workers)
        return list(itertools.chain.from_iterable(itertools.islice(iter(dl), 40 - skip)))

    full = stream()                                     # 40 batches = 160 pages, ~3.5 passes over the corpus
    resumed = stream(skip=13)                           # a run killed after 13 batches and resumed
    assert resumed == full[13 * 4:]                     # continues with the same sequence, nothing replayed
    assert stream(skip=0)[:8] == full[:8] and full[:8] != resumed[:8]


def test_param_groups_exclude_biases_norms_and_embeddings_from_weight_decay():
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    groups = model.layerwise_param_groups(1e-4, layer_decay=0.5, weight_decay=0.05, new_lr=1e-3)
    wd = {n: g["weight_decay"] for g in groups for n in g["names"]}
    lr = {n: g["lr"] for g in groups for n in g["names"]}
    assert len(wd) == sum(1 for _ in model.parameters())                  # every parameter exactly once
    no_decay = {n for n, v in wd.items() if v == 0.0}
    for n in ("layoutlmv3.cls_token", "layoutlmv3.pos_embed", "layoutlmv3.image_mask_token",
              "layoutlmv3.embeddings.word_embeddings.weight", "layoutlmv3.embeddings.position_embeddings.weight",
              "layoutlmv3.embeddings.x_position_embeddings.weight", "layoutlmv3.embeddings.token_type_embeddings.weight",
              "layoutlmv3.embeddings.LayerNorm.weight", "layoutlmv3.LayerNorm.weight", "layoutlmv3.norm.weight",
              "layoutlmv3.encoder.layer.0.attention.output.LayerNorm.weight",
              "layoutlmv3.encoder.layer.1.attention.self.query.bias", "layoutlmv3.patch_embed.proj.bias",
              "layoutlmv3.encoder.rel_pos_bias.weight", "layoutlmv3.encoder.rel_pos_x_bias.weight",
              "mlm_head.decoder.bias", "mim_head.0.weight", "wpa_head.net.0.bias"):
        assert n in no_decay, n
    decay = set(wd) - no_decay
    assert decay == {n for n, p in model.named_parameters() if p.ndim >= 2 and not n.startswith(
        ("layoutlmv3.cls_token", "layoutlmv3.pos_embed", "layoutlmv3.image_mask_token", "layoutlmv3.embeddings.",
         "layoutlmv3.encoder.rel_pos"))}
    assert {"layoutlmv3.patch_embed.proj.weight", "layoutlmv3.encoder.layer.0.attention.self.query.weight",
            "layoutlmv3.encoder.layer.1.output.dense.weight", "mlm_head.dense.weight", "mim_head.1.weight",
            "wpa_head.net.2.weight"} <= decay and all(wd[n] == 0.05 for n in decay)
    # layer-wise lr decay is unchanged by the split: both halves of a depth have the same lr
    assert lr["layoutlmv3.encoder.layer.0.attention.self.query.weight"] == pytest.approx(1e-4 * 0.5 ** 2)
    assert lr["layoutlmv3.encoder.layer.0.attention.self.query.bias"] == pytest.approx(1e-4 * 0.5 ** 2)
    assert lr["layoutlmv3.embeddings.word_embeddings.weight"] == lr["layoutlmv3.patch_embed.proj.weight"]
    assert lr["mim_head.1.weight"] == lr["mim_head.1.bias"] == pytest.approx(1e-3)
