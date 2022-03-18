# [EdgeFormer](https://arxiv.org/abs/2202.07959)

**EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation** 

[EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation](https://arxiv.org/abs/2202.07959). Tao Ge and Furu Wei

- March 2022: release code and pretrained checkpoints.

---

## Pretrained Models

- [EdgeFormer (Adapter-LA)](https://msranlp.blob.core.windows.net/edgeformer/v1/edgeformer_lora32_pretrain_checkpoint_250k.pt): #enc-dec=12-2; #hidden=512; #head=8; #enc-FFN=2048, #dec-FFN=128, #LoRA-r=32 (#parameters: 11M)
- [Vocabulary](https://msranlp.blob.core.windows.net/edgeformer/v1/dict.src.txt) and [Sentencepiece-model](https://msranlp.blob.core.windows.net/edgeformer/v1/spm2k-fy22.model)
- EdgeFormer can be finetuned to support seq2seq generation in English (by now).


## Downstream seq2seq tasks

We evaluate EdgeFormer on the benchmarks of three popular seq2seq tasks: CoNLL-14 for GEC, XSUM for Abstractive Summarization, and SQuAD-NQG for Question Generation.

[**CoNLL-14**](https://aclanthology.org/W14-1701.pdf)

|   Model   |   #Params   |#FLOPS|F0.5|
|-----------|-------------|-----------|-----------|
| [Transformer-base](https://arxiv.org/abs/1706.03762)     | 44M |     1.8G    | 50.1 |
| Pretrained 12+2 [Universal Transformer](https://arxiv.org/abs/1807.03819)      | 7.4M        | 1.4G      | 51.3 |
| Pretrained 12+2 Universal Transformer (wide)      | 9.4M        | 1.9G      | 51.7 |
| Pretrained EdgeFormer   | 9.4M        | 1.3G      | **52.7**      |

[**XSUM**](https://arxiv.org/pdf/1808.08745.pdf)

|   Model   |   #Params   |#FLOPS|ROUGE-1|ROUGE-2|ROUGE-L|
|-----------|-------------|-----------|-----------|-----------|-----------|
| Transformer-base     | 44M |     1.8G    | 31.2 | 10.7 | 24.9 |
| Pretrained 12+2 Universal Transformer   | 7.4M        | 1.4G      | 34.4 | 13.4 | 27.9 |
| Pretrained 12+2 Universal Transformer (wide)       | 9.4M        | 1.9G      | 35.1 | 14.0 | 28.6 |
| Pretrained EdgeFormer   | 9.4M        | 1.3G      | **36.3**      | **14.8** | **29.5** |

[**SQuAD-NQG**](https://arxiv.org/abs/1705.00106)

|   Model   |   #Params   |#FLOPS|B4|MTR|ROUGE-L|
|-----------|-------------|-----------|-----------|-----------|-----------|
| Transformer-base    | 44M |     1.8G    | 2.6 | 9.0 | 26.0|
| Pretrained 12+2 Universal Transformer      | 7.4M        | 1.4G      | 18.3 | 21.0 | 45.9 |
| Pretrained 12+2 Universal Transformer (wide)       | 9.4M        | 1.9G      | 18.7 | 21.3 | 46.1 |
| Pretrained EdgeFormer   | 9.4M        | 1.3G      | **19.0**      | **21.7** | **46.3** |


## Setup

pip install --editable ./
```

## Fine-tuning
```bash
PRETRAINED_MODEL=/path/to/checkpoint/model.pt
fairseq-train /path/to/binarized/data \
        --restore-file $PRETRAINED_MODEL  --reset-lr-scheduler --reset-optimizer --reset-dataloader \
        --task translation \
        --criterion label_smoothed_cross_entropy \
        --arch transformer_edge --encoder-layers 12 --decoder-ffn-embed-dim 128 --lora-r 32 --lora-r-shape 0 \
        --share-all-embeddings \
        --required-batch-size-multiple 8 \
        --optimizer adam \
        --adam-betas '(0.9,0.98)' \
        --adam-eps 1e-6 \
        --clip-norm 1.0 \
        --lr-scheduler polynomial_decay \
        --lr 0.00015 \
        --warmup-updates 8000 \
        --total-num-update 100000 \
        --max-update 100000 --max-epoch 1000 \
        --max-tokens 20000 \
        --update-freq 1 \
        --log-format simple \
        --log-interval 1000 \
        --save-interval-updates 5000 \
        --fp16 \
        --fp16-init-scale 4 \
        --fp16-scale-window 256 \
        --min-loss-scale 0.0001 \
        --seed 1
        --save-dir /path/to/save/checkpoints
        --ddp-backend legacy_ddp
```
**Note: 
- Please adjust the hyperparameters like `lr` and `warmup-updates` based on the datasets and tasks.
- Please adjust the `max-tokens` and `update-freq` to suit in different experimental environments.
- Use `--fp16` for more efficient training on the devices that have Tensor Cores.

5. Evaluation:

```bash
fairseq-generate $data_bin \
    --path $save_dir/checkpoint_best.pt \
    --batch-size 64 --beam 5 --remove-bpe=sentencepiece
```

---

## Citation

If you find this repository useful, please consider citing our work:
```
@article{ge2022edgeformer,
  title={EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation},
  author={Ge, Tao and Wei, Furu},
  journal={arXiv preprint arXiv:2202.07959},
  year={2022}
}
```

## Acknowledgement

This repository is built using the [Fairseq](https://github.com/pytorch/fairseq) repository.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using EdgeFormer models, please submit a GitHub issue.

For other communications related to EdgeFormer, please contact [Tao Ge](https://www.microsoft.com/en-us/research/people/tage/) (`tage@microsoft.com`), [Furu Wei](http://gitnlp.org/) (`fuwei@microsoft.com`).
