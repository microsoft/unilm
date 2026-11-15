# LayoutLMv3 on whole document pages (more than 512 tokens)

The released LayoutLMv3 checkpoints have 512 text positions, so a page with more text is cut off and the rest is never labelled ([#750](https://github.com/microsoft/unilm/issues/750), [#942](https://github.com/microsoft/unilm/issues/942)). This folder fine-tunes LayoutLMv3 on whole pages of up to 2,048 or 4,096 tokens. It needs no extra pre-training and makes two small changes to Hugging Face `transformers`' `LayoutLMv3ForTokenClassification`:

1. **Longer positions** (`extend_position_embeddings`). The 512 learned position embeddings are repeated: position 512 + i reuses the vector of position i, as Longformer did to extend RoBERTa. The relative position biases are bucketed by distance and need no change.
2. **Low-memory exact attention** (`use_lut_attention`). The 1D and 2D relative position biases depend only on the bucketed distance between two tokens. The bias is therefore gathered once per forward pass from small per-distance tables, in the compute dtype, and shared by all layers. It is passed to PyTorch's `scaled_dot_product_attention` as the additive mask. Outputs match `transformers`' attention (tested), and the memory-efficient SDPA kernel also runs on V100 GPUs.

## Results

The test pages are long: cutting them at 512 tokens hides 41% to 58% of the labelled words. Every test word is scored, and words the model never sees count as predicted "O". Results are entity F1 for LayoutLMv3-base, mean of 5 seeds, 2,000 steps, batch 2, learning rate 3e-5.

| Dataset | Cut at 512 | Whole page |
|---|---:|---:|
| [VRDU](https://github.com/google-research-datasets/vrdu) Ad-buy forms (TV ad orders) | 68.31 | **92.52** (2,048) |
| VRDU Registration forms (US filings) | 63.96 | **87.82** (1,024) |
| [RealKIE](https://arxiv.org/abs/2403.20101) FCC invoices | 42.46 | **67.57** (2,048) |

- Whole pages are better on every dataset and in every seed (paired p < 1e-5).
- Reading 512-token windows and merging the predictions per word also works well: 90.97, 88.65 and 66.39. One whole-page pass is 1.2 to 1.6 points better on Ad-buy and RealKIE (p ≈ 0.01) and ties on Registration.
- Fine-tuning learns as fast as before. On Ad-buy, the whole-page model passes the 512-token model's final score after 500 of the 2,000 steps.
- Interpolating the position embeddings instead of repeating them was 0.7 points worse, mostly on pages of 1,024 tokens or fewer.

Inference memory for one page, fp16, LayoutLMv3-base (197 image tokens added):

| Text tokens | 512 | 2,048 | 4,096 |
|---|---:|---:|---:|
| `transformers` attention | 0.81 GB | 2.06 GB | 5.77 GB |
| `use_lut_attention` | 0.69 GB | 0.80 GB | 1.57 GB |

Fine-tuning at 2,048 tokens (batch 2, fp16, gradient checkpointing) peaks at 4.0 GB, and at 4,096 tokens at 6.7 GB. Batches are padded only to their longest page, so short pages cost the same as before.

These numbers were measured on an A100 limited to 15 GB and to kernels a V100 supports, not on a V100. Each result comes from a run logged in the [layoutlmv3-pretraining](https://github.com/14H034160212/layoutlmv3-pretraining) repository (README, "Long documents within a 16 GB GPU").

## Usage

```bash
pip install "transformers>=4.46,<5" datasets seqeval pymupdf
```

Prepare a dataset of pages (words, 0-1000 boxes, BIO tags and page images, saved with `datasets.save_to_disk`):

```bash
# VRDU Ad-buy forms: dataset.jsonl.gz, meta.json and pdfs/ from google-research-datasets/vrdu (ad-buy-form/main)
python prepare_vrdu.py --vrdu_dir vrdu/ad-buy-form/main \
    --split ../few_shot-splits/DeepForm-mixed_template-train_200-test_300-valid_100-SD_0.json --out data/vrdu_adbuy

# RealKIE FCC invoices (CC BY-NC 4.0)
python prepare_realkie.py --realkie_dir realkie --dataset fcc_invoices --out data/realkie_fcc_invoices
```

Fine-tune and evaluate:

```bash
python finetune.py --data data/vrdu_adbuy --max_length 512    # baseline: pages cut at 512 tokens
python finetune.py --data data/vrdu_adbuy --max_length 2048   # whole pages
```

On your own model:

```python
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from long_documents import extend_position_embeddings, use_lut_attention

model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=num_labels)
model = use_lut_attention(extend_position_embeddings(model, 2048))
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
encoding = processor(image, words, boxes=boxes, word_labels=labels, truncation=True, max_length=2048)
```

## Files

- `long_documents.py`: `extend_position_embeddings` and `use_lut_attention`
- `finetune.py`: token-classification fine-tuning and strict entity F1
- `prepare_vrdu.py`, `prepare_realkie.py`: page-level datasets from VRDU and RealKIE
- `test_long_documents.py`: CPU tests (`python -m pytest test_long_documents.py`)

## Notes

- `use_lut_attention` does not support `output_attentions` or `head_mask`.
- As in `transformers` since 4.42, the relative position bias tables are not trained ([huggingface/transformers#50636](https://github.com/huggingface/transformers/issues/50636)).
- The copy initialisation of positions is also proposed for all encoders with learned absolute positions in [huggingface/transformers#50769](https://github.com/huggingface/transformers/issues/50769).
