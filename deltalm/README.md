# [DeltaLM](https://arxiv.org/abs/2106.13736)

**Encoder-Decoder Pre-training for Language Generation and Translation** 

[DeltaLM: Encoder-Decoder Pre-training for Language Generation and Translation by Augmenting Pretrained Multilingual Encoders.](https://arxiv.org/abs/2106.13736) Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, Alexandre Muzio, Saksham Singhal, Hany Hassan Awadalla, Xia Song, Furu Wei. CoRR abs/2106.13736.

[mT6: Multilingual Pretrained Text-to-Text Transformer with Translation Pairs.](https://arxiv.org/abs/2104.08692) Zewen Chi, Li Dong, Shuming Ma, Shaohan Huang, Xian-Ling Mao, Heyan Huang, and Furu Wei. In EMNLP 2021.

- September 2021: DeltaLM ranks first on the [WMT21 multilingual translation task](http://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html).
- August 2021: release code and pretrained checkpoints.

---

## Pretrained Models

- [DeltaLM-base](https://deltalm.blob.core.windows.net/deltalm/deltalm-base.pt): #enc-dec=12-6; #hidden=768; #head=12; #FFN=3072 (#parameters: 360M)
- [DeltaLM-large](https://deltalm.blob.core.windows.net/deltalm/deltalm-large.pt): #enc-dec=24-12; #hidden=1024; #head=16; #FFN=4096 (#parameters: 830M)
- [Vocabulary](https://deltalm.blob.core.windows.net/deltalm/dict.txt) and [Sentencepiece-model](https://deltalm.blob.core.windows.net/deltalm/spm.model)
- DeltaLM can be finetuned to support language generation and translation tasks for **100+ languages**


## Cross-lingual Abstractive Summarization - [Wikilingua](https://arxiv.org/abs/2010.03093)

We evaluate DeltaLM on cross-lingual abstractive summarization benchmark. We report the results by averaging the numbers in different languages. 

|   Model   |   #Params   |  ROUGE-1  |  ROUGE-2  |  ROUGE-L  |
|-----------|-------------|-----------|-----------|-----------|
| [mBART](https://arxiv.org/abs/2001.08210)     | 610M        | 34.5      | 12.9      | **28.7**      |
| [mT5](https://arxiv.org/abs/2010.11934)       | 300M        | 27.5      | 8.8       | 22.8      |
| [mT5](https://arxiv.org/abs/2010.11934)       | 580M        | 31.8      | 11.5      | 26.0      |
| DeltaLM   | 360M        | **35.3**      | **13.4**      | **28.7**      |


## Setup

```bash
git submodule update --init deltalm/fairseq
cd deltalm/
pip install --editable fairseq/
```

## Fine-tuning

1. Organize the raw data in the following structure:
```
.
+-- /path/to/data/
|   +-- train.src
|   +-- train.tgt
|   +-- valid.src
|   +-- valid.tgt
```

*Examples (IWSLT14 German to English)*:
```bash
bash examples/prepare_iwslt14.sh /tmp/iwslt14
```

2. Tokenize the data using [Sentencepiece](https://github.com/google/sentencepiece):

```bash
spm_encode --model=/path/to/checkpoint/spm.model --output_format=piece < train.src > train.spm.src
spm_encode --model=/path/to/checkpoint/spm.model --output_format=piece < train.tgt > train.spm.tgt
spm_encode --model=/path/to/checkpoint/spm.model --output_format=piece < valid.src > valid.spm.src
spm_encode --model=/path/to/checkpoint/spm.model --output_format=piece < valid.tgt > valid.spm.tgt
spm_encode --model=/path/to/checkpoint/spm.model --output_format=piece < test.src > test.spm.src
spm_encode --model=/path/to/checkpoint/spm.model --output_format=piece < test.tgt > test.spm.tgt
```

*Examples (IWSLT14 German to English)*:
```bash
bash examples/binary_iwslt14.sh \
     /tmp/iwslt14/iwslt14.tokenized.de-en \
     /tmp/iwslt14/iwslt14.spm \
     /path/to/checkpoint/spm.model
```

3. Binary the data:

```bash
data_bin=/path/to/data-bin/
python preprocess.py  \
    --trainpref train.spm \
    --validpref valid.spm \
    --testpref test.spm \
    --source-lang src --target-lang tgt \
    --destdir $data_bin \
    --srcdict /path/to/checkpoint/dict.txt \
    --tgtdict /path/to/checkpoint/dict.txt \
    --workers 40
```

*Examples (IWSLT14 German to English)*:
```bash
bash examples/binary_iwslt14.sh \
     /tmp/iwslt14/iwslt14.spm \
     /tmp/iwslt14/iwslt14.bin \
     /path/to/checkpoint/dict.txt
```

4. Fine-tuning:

```bash
PRETRAINED_MODEL=/path/to/checkpoint/model.pt
python train.py $data_bin \
    --save-dir $save_dir \
    --arch deltalm_base \
    --pretrained-deltalm-checkpoint $PRETRAINED_MODEL \
    --share-all-embeddings \
    --max-source-positions 512 --max-target-positions 512 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr $lr \
    --warmup-init-lr 1e-07 \
    --stop-min-lr 1e-09 \
    --warmup-updates 4000 \
    --max-update 400000 \
    --max-epoch 100 \
    --max-tokens $batch_size \
    --update-freq 1 \
    --seed 1 \
    --log-format simple \
    --skip-invalid-size-inputs-valid-test
```
**Note: 
- For large checkpoint, please set `--arch deltalm_large`.
- Please adjust the `max-tokens` and `update-freq` to suit in different experimental environments. Recommendation of the total batch size is `4096 * 128` tokens per step.
- Use `--fp16` for more efficient training on the devices that have Tensor Cores.

*Examples (IWSLT14 German to English)*:
```bash
bash examples/train_iwslt14.sh \
     /tmp/iwslt14/iwslt14.bin \
     /tmp/iwslt14/checkpoints \
     /path/to/checkpoint/model.pt
```

5. Evaluation:

```bash
python generate.py $data_bin \
    --path $save_dir/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe=sentencepiece
```

*Examples (IWSLT14 German to English)*:
```bash
bash examples/evaluate_iwslt14.sh \
     /tmp/iwslt14/iwslt14.bin \
     /tmp/iwslt14/checkpoints
```

---

## Citation

If you find this repository useful, please consider citing our work:
```
@article{deltalm,
      title={{DeltaLM}: Encoder-Decoder Pre-training for Language Generation and Translation by Augmenting Pretrained Multilingual Encoders}, 
      author={Shuming Ma and Li Dong and Shaohan Huang and Dongdong Zhang and Alexandre Muzio and Saksham Singhal and Hany Hassan Awadalla and Xia Song and Furu Wei},
      year={2021},
      eprint={2106.13736},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement

This repository is built using the [Fairseq](https://github.com/pytorch/fairseq) repository.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using DeltaLM models, please submit a GitHub issue.

For other communications related to DeltaLM, please contact Shuming Ma (`shumma@microsoft.com`), [Furu Wei](http://gitnlp.org/) (`fuwei@microsoft.com`).
