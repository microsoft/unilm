# Generalized Aggressive Decoding

Codes (originally from https://github.com/hemingkx/GAD) for Generalized Aggressive Decoding that is originally proposed in the paper [Lossless Speedup of Autoregressive Translation with Generalized Aggressive Decoding](https://arxiv.org/pdf/2203.16487.pdf).

### Download model

| Description | Model                                                        |
| ----------- | ------------------------------------------------------------ |
| wmt14.en-de | [at-verifier-base](https://drive.google.com/file/d/1L9z0Y5rked_tYn7Fllh-0VsRdgBHN1Mp/view?usp=sharing)， [nat-drafter-base (k=25)](https://drive.google.com/file/d/1fPYt1QGgIrNfk78XvGnrx_TeDRYePr2e/view?usp=sharing) |
| wmt14.de-en | [at-verifier-base](https://drive.google.com/file/d/1h5EdTEt2PMqvAqCq2G5bRhCeWk8LzwoG/view?usp=sharing)， [nat-drafter-base (k=25)](https://drive.google.com/file/d/1IEX2K65rgv5SUHWxiowXYaS--Zqr3GvT/view?usp=sharing) |
| wmt16.en-ro | [at-verifier-base](https://drive.google.com/file/d/1WocmZ9iw_OokYZY_BtzNAjGsgRXB-Aft/view?usp=sharing)， [nat-drafter-base (k=25)](https://drive.google.com/file/d/1V_WbPRbgmIy-4oZDkws9mdFSw8n8KOGm/view?usp=sharing) |
| wmt16.ro-en | [at-verifier-base](https://drive.google.com/file/d/1LWHC56HvTtvs58EMwoYMT6jKByuMW1dB/view?usp=sharing)， [nat-drafter-base (k=25)](https://drive.google.com/file/d/1P21nU3u4WdJueEl4nqAY-cwUKAvzPu8A/view?usp=sharing) |

### Requirements

- Python >= 3.7
- Pytorch >= 1.5.0

### Installation

```
conda create -n gad python=3.7
cd GAD
pip install --editable .
```

### Preprocess

We release the bpe codes and our dict in `./data`.

```
text=PATH_YOUR_DATA
src=source_language
tgt=target_language

model_path=PATH_TO_MODEL_DICT_DIR

fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref $text/train --validpref $text/valid --testpref $text/test \
    --destdir PATH_TO_BIN_DIR --workers 60 \
    --srcdict ${model_path}/dict.${src}.txt \
    --tgtdict ${model_path}/dict.${tgt}.txt
```

### Train

For training the NAT drafter of GAD (check `train.sh`)

```
python train.py ${bin_path} --arch block --noise block_mask --share-all-embeddings \
    --criterion glat_loss --label-smoothing 0.1 --lr ${lr} --warmup-init-lr 1e-7 \
    --stop-min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates ${warmup} \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 \
    --task translation_lev_modified --max-tokens ${max_tokens} --weight-decay 0.01 \
    --dropout ${dropout} --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 \
    --decoder-embed-dim 512 --fp16 --max-source-positions 1000 \
    --max-target-positions 1000 --max-update ${update} --seed ${seed} --clip-norm 5 \
    --save-dir ./checkpoints --src-embedding-copy --log-interval 1000 \
    --user-dir block_plugins --block-size ${size} --total-up ${update} \
    --update-freq ${update_freq} --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu
```

### Inference

For GAD++   (check `inference.sh`, set `beta=1` for vanilla GAD):

```
python inference.py ${data_dir} --path ${checkpoint_path} --user-dir block_plugins \
    --task translation_lev_modified --remove-bpe --max-sentences 20 \
    --source-lang ${src} --target-lang ${tgt} --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test \
    --AR-path ${AR_checkpoint_path} --input-path ${input_path} \
    --output-path ${output_path} --block-size ${block_size} --beta ${beta} --tau ${tau} \
    --batch ${batch} --beam ${beam} --strategy ${strategy}
```

> We test the inference latency of GAD with batch 1 implementation, check `inference_paper.py` for details.
>

Calculating compound split bleu:

```
./ref.sh
```

### Note

This code is based on GLAT [(https://github.com/FLC777/GLAT)](https://github.com/FLC777/GLAT). 

