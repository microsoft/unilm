# SpeechT5

<!--**Pre-trained models for speech related tasks**-->

 [**SpeechT5**](https://arxiv.org/abs/2110.07205): **Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing**

Official PyTorch implementation and pretrained models of SpeechT5

- Oct 2021: release preprint in [arXiv](https://arxiv.org/abs/2110.07205)
- Feb 2022: accepted by [ACL 2022](https://www.2022.aclweb.org/)


## Pre-Trained Models

|  Model   |               Pre-training Dataset               | Fine-tuning Dataset | Model |
| :------: | :----------------------------------------------: | :-----------------: | :-----: |
| SpeechT5 Base | [960 hrs LibriSpeech](http://www.openslr.org/12) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) |          -          | [HuggingFace](https://huggingface.co/ajyy/SpeechT5/resolve/main/speecht5_base.pt)<br /> [Google Drive](https://drive.google.com/file/d/1Sq00uZ1pw6Z4OUaqhOWzQEJxIVWgAO5U/view?usp=sharing)  |
| SpeechT5 Base | [960 hrs LibriSpeech](http://www.openslr.org/12) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) | [100 hrs LibriSpeech](http://www.openslr.org/12) | [HuggingFace](https://huggingface.co/ajyy/SpeechT5/resolve/main/speecht5_base_asr.pt)<br /> [Google Drive](https://drive.google.com/file/d/1qLKJ81JPWOGf1MHfjSmgtZyqqTqgI6kT/view?usp=sharing)  |
| SpeechT5 Large | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) |          -          | [Google Drive](https://drive.google.com/file/d/1M79b1jetSPOVxWVMIX-y0URvDjNskZKp/view?usp=sharing)  |

## Language Model and Vocabulary
|  Model   |  Dataset | Model | Vocabulary | SPM Model |
| :------: | :------: | :---: | :--------: | :-------: |
| LM | [LibriSpeech LM Dataset](https://www.openslr.org/11/) | [LM Model](https://drive.google.com/uc?export=download&id=1y0TGnKAMKUW5C8l8yrvGjh9RRZETPdv7)  | [Vocabulary](https://drive.google.com/uc?export=download&id=19hcQ58RHZ6CssxF8Qp6yEF1NW_AXxObK) | [SPM Model](https://drive.google.com/uc?export=download&id=1wClgQjXXoU2lmpbaEa1v2SqMbg7cAutq) |


## Setup
```
git submodule update --init SpeechT5/fairseq
cd SpeechT5/
pip install --editable fairseq/
pip install espnet
```

## Data Preparation

### Speech data and S2T Data
Please follow the steps for preparing wav2vec 2.0 manifest in [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) and preparing HuBERT label in [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans).

We add a third column for the speaker embedding, which is provided in [here](https://drive.google.com/uc?export=download&id=16QOUURZBrW7-GYbVG_gXt3mTMlZmQoH0).
It includes the speaker embeddings for 960hr training data and dev-other data of LibriSpeech.

We also provide example manifests for your reference in [here](https://drive.google.com/drive/folders/1Ja08XjOHe6vP8lZtLVrJM8173aPQCR_y?usp=sharing).

### Text Data
Please use [fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess) to generate the index and bin files of the text data. Note that we use sentencepiece to pre-process the text, so please refer to [here](https://github.com/microsoft/SpeechT5#language-model-and-vocabulary) to download the SPM model and dictionary for preparing text data. This means you firstly need to use the SPM model to process the text and then use [fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess) with the provided dictionary to get the index and bin files.

## Load Pre-Trained Models

```
import torch
from speecht5.tasks.speecht5 import SpeechT5Task
from speecht5.models.speecht5 import T5TransformerModel

checkpoint = torch.load('/path/to/speecht5_checkpoint')

checkpoint['cfg']['task'].t5_task = 'pretrain'
checkpoint['cfg']['task'].hubert_label_dir = "/path/to/hubert_label"
checkpoint['cfg']['task'].data = "/path/to/tsv_file"

task = SpeechT5Task.setup_task(checkpoint['cfg']['task'])
model = T5TransformerModel.build_model(checkpoint['cfg']['model'], task)
model.load_state_dict(checkpoint['model'])
```

## Pre-Training

### 960hr LibriSpeech + LibriSpeech-LM

```
DATA_ROOT=
SAVE_DIR=
LABEL_DIR=
TRAIN_SET="speech_train|text_train"
VALID_SET="speech_valid|text_valid"


fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 32 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir SpeechT5/speecht5 \
  --log-format json \
  --seed 1337 \
  --fp16 \
  \
  --task speecht5 \
  --t5-task pretrain \
  --label-rates 50 \
  --sample-rate 16000 \
  --random-crop \
  \
  --num-workers 0 \
  --max-tokens 1400000 \
  --max-speech-sample-size 250000 \
  --update-freq 2 \
  --batch-ratio "[1,0.0086]" \
  \
  --criterion speecht5 \
  --optimizer adam \
  --reset-optimizer \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-06 \
  --weight-decay 0.01 \
  --power 1 \
  --clip-norm 5.0 \
  --lr 0.0002 \
  --lr-scheduler polynomial_decay \
  \
  --max-update 800000 \
  --warmup-updates 64000 \
  --total-num-update 800000 \
  --save-interval-updates 3000 \
  --skip-invalid-size-inputs-valid-test \
  --required-batch-size-multiple 1 \
  \
  --arch t5_transformer_base \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --use-codebook \
  --codebook-prob 0.1 \
  --loss-weights="[10,0.1]" \
  --max-text-positions 600 \
```

## Finetune

### ASR

#### Training

```
DATA_ROOT=
SAVE_DIR=
TRAIN_SET=
VALID_SET=
LABEL_DIR=
BPE_TOKENIZER=
USER_DIR=
PT_CHECKPOINT_PATH=

mkdir -p ${SAVE_DIR}
fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 8 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  --fp16 \
  \
  --task speecht5 \
  --t5-task s2t \
  --sample-rate 16000 \
  --num-workers 0 \
  --max-tokens 1600000 \
  --update-freq 2 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  \
  --criterion speecht5 \
  --report-accuracy \
  --zero-infinity \
  --ce-weight 0.5 \
  --ctc-weight 0.5 \
  --sentence-avg \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-08 \
  --weight-decay 0.1 \
  --clip-norm 25.0 \
  --lr 0.00006 \
  --lr-scheduler tri_stage \
  --phase-ratio "[0.1, 0.4, 0.5]" \
  --final-lr-scale 0.05 \
  \
  --max-update 80000 \
  --max-text-positions 600 \
  --required-batch-size-multiple 1 \
  --save-interval-updates 3000 \
  --skip-invalid-size-inputs-valid-test \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 13000 \
  \
  --keep-last-epochs 10 \
  --feature-grad-mult 1.0 \
  --best-checkpoint-metric s2t_accuracy \
  --maximize-best-checkpoint-metric \
  --finetune-from-model ${PT_CHECKPOINT_PATH}
```

#### Inference
Note that joint CTC/Decoder inference is only supported when batch size is 1.

```
CHECKPOINT_PATH=
DATA_ROOT=
SUBSET=
BPE_TOKENIZER=
LABEL_DIR=
USER_DIR=
BEAM=
MAX_TOKENS=
CTC_WEIGHT=
LM_WEIGHT=
LM_PATH=

fairseq-generate ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --user-dir ${USER_DIR} \
  --task speecht5 \
  --t5-task s2t \
  --path ${CHECKPOINT_PATH} \
  --hubert-label-dir ${LABEL_DIR} \
  --ctc-weight ${CTC_WEIGHT} \
  --lm-weight ${LM_WEIGHT} \
  --lm-path ${LM_PATH} \
  --max-tokens ${MAX_TOKENS} \
  --beam ${BEAM} \
  --scoring wer \
  --max-len-a 0 \
  --max-len-b 620 \
  --sample-rate 16000
```

### TTS

The manifest and pre-trained vocoder can be found in [huggingface](https://huggingface.co/mechanicalsea/speecht5-tts), which may be helpful to reproduce the results of SpeechT5 TTS model.

We also provide re-implementation of TTS fine-tuned model [speecht5_tts.pt](https://huggingface.co/mechanicalsea/speecht5-tts/blob/main/speecht5_tts.pt), but with a smaller batch size or max updates, which can be helpful.

#### Training

```
DATA_ROOT=
SAVE_DIR=
TRAIN_SET=
VALID_SET=
LABEL_DIR=
BPE_TOKENIZER=
USER_DIR=
PT_CHECKPOINT_PATH=

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 8 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  --fp16 \
  \
  --task speecht5 \
  --t5-task t2s \
  --sample-rate 16000 \
  --num-workers 4 \
  --max-tokens 3200000 \
  --update-freq 1 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --max-tokens-valid 3200000 \
  \
  --criterion speecht5 \
  --use-guided-attn-loss \
  --report-accuracy \
  --sentence-avg \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --dropout 0.15 \
  --activation-dropout 0.15 \
  --attention-dropout 0.15 \
  --encoder-layerdrop 0.0 \
  --decoder-layerdrop 0.0 \
  --weight-decay 0.0 \
  --clip-norm 25.0 \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 \
  --feature-grad-mult 1.0 \
  \
  --max-update 120000 \
  --max-text-positions 600 \
  --min-speech-sample-size 1056 \
  --max-speech-sample-size 480256 \
  --max-speech-positions 1876 \
  --required-batch-size-multiple 1 \
  --skip-invalid-size-inputs-valid-test \
  --keep-last-epochs 10 \
  --validate-after-updates 20000 \
  --validate-interval 50 \
  --log-interval 10 \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 20000 \
  \
  --finetune-from-model ${PT_CHECKPOINT_PATH}
```

#### Inference

Generating speech is available only if batch size is 1.

```
SPEECHT5_CODE_DIR=
CHECKPOINT_PATH=
DATA_ROOT=
SUBSET=
BPE_TOKENIZER=
LABEL_DIR=
USER_DIR=
RESULTS_PATH=

python3 ${SPEECHT5_CODE_DIR}/SpeechT5/scripts/generate_speech.py ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --user-dir ${USER_DIR} \
  --task speecht5 \
  --t5-task t2s \
  --path ${CHECKPOINT_PATH} \
  --hubert-label-dir ${LABEL_DIR} \
  --batch-size 1 \
  --results-path ${RESULTS_PATH} \
  --sample-rate 16000
```

### ST

Here we follow [fairseq/speech_to_text/mustc](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md#data-preparation) to generate vocabulary, which is different from the pre-trained models. So we randomly initilize the embedding table of the pre-trained models during fine-tuning.

#### Training

```
DATA_ROOT=
SAVE_DIR=
TRAIN_SET=
VALID_SET=
LABEL_DIR=
BPE_TOKENIZER=
USER_DIR=
PT_CHECKPOINT_PATH=

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 8 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  --fp16 \
  \
  --task speecht5 \
  --t5-task s2t \
  --sample-rate 16000 \
  --num-workers 6 \
  --max-tokens 480256 \
  --update-freq 4 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --max-tokens-valid 3200000 \
  \
  --criterion speecht5 \
  --label-smoothing 0.1 \
  --report-accuracy \
  --sentence-avg \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --weight-decay 0.0 \
  --clip-norm 10.0 \
  --lr 0.0002 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 25000 \
  --feature-grad-mult 1.0 \
  \
  --max-update 80000 \
  --max-text-positions 600 \
  --min-speech-sample-size 1056 \
  --max-speech-sample-size 480256 \
  --max-speech-positions 1876 \
  --required-batch-size-multiple 1 \
  --skip-invalid-size-inputs-valid-test \
  --keep-last-epochs 10 \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 0 \
  --mask-prob 0.5 \
  --mask-channel-prob 0.5 \
  \
  --finetune-from-model ${PT_CHECKPOINT_PATH}
```

#### Inference

```
FAIRSEQ_DIR=
CHECKPOINT_PATH=
DATA_ROOT=
BPE_TOKENIZER=
LABEL_DIR=
USER_DIR=
MAX_TOKENS=

python3 ${FAIRSEQ_DIR}/scripts/average_checkpoints.py \
  --inputs ${CHECKPOINT_PATH} \
  --num-epoch-checkpoints 10 \
  --output ${CHECKPOINT_PATH}/avg_last_10_checkpoint.pt

fairseq-generate ${DATA_ROOT} \
        --gen-subset tst-COMMON \
        --bpe-tokenizer ${BPE_TOKENIZER} \
        --user-dir ${USER_DIR} \
        --task speecht5 \
        --t5-task s2t \
        --path ${CHECKPOINT_PATH}/avg_last_10_checkpoint.pt \
        --hubert-label-dir ${LABEL_DIR} \
        --max-tokens ${MAX_TOKENS} \
        --min-speech-sample-size 1056 \
        --beam 5 \
        --scoring sacrebleu \
        --max-len-a 0 \
        --max-len-b 620 \
        --sample-rate 16000
```

### VC

The manifest and pre-trained vocoder can be found in [huggingface](https://huggingface.co/mechanicalsea/speecht5-vc), which may be helpful to reproduce the results of SpeechT5 VC model.

We also provide re-implementation of VC fine-tuned model [speecht5_vc.pt](https://huggingface.co/mechanicalsea/speecht5-vc/blob/main/speecht5_vc.pt), but with a smaller batch size or max updates, which can be helpful.

#### Training


```
DATA_ROOT=
SAVE_DIR=
TRAIN_SET=
VALID_SET=
LABEL_DIR=
BPE_TOKENIZER=
USER_DIR=
PT_CHECKPOINT_PATH=

fairseq-train ${DATA_ROOT} \
        --save-dir ${SAVE_DIR} \
        --tensorboard-logdir ${SAVE_DIR} \
        --train-subset ${TRAIN_SET} \
        --valid-subset ${VALID_SET} \
        --hubert-label-dir ${LABEL_DIR} \
        --distributed-world-size 8 \
        --distributed-port 0 \
        --ddp-backend legacy_ddp \
        --user-dir ${USER_DIR} \
        --log-format json \
        --seed 1 \
        --fp16 \
        \
        --task speecht5 \
        --t5-task s2s \
        --sample-rate 16000 \
        --num-workers 4 \
        --max-tokens 1280000 \
        --update-freq 3 \
        --max-tokens-valid 1280000 \
        \
        --criterion speecht5 \
        --use-guided-attn-loss \
        --report-accuracy \
        --sentence-avg \
        \
        --optimizer adam \
        --dropout 0.2 \
        --activation-dropout 0.2 \
        --attention-dropout 0.2 \
        --encoder-layerdrop 0.05 \
        --decoder-layerdrop 0.0 \
        --clip-norm 1.0 \
        --lr 0.0001 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 6000 \
        --feature-grad-mult 1.0 \
        \
        --max-update 60000 \
        --max-text-positions 600 \
        --min-speech-sample-size 1056 \
        --max-speech-sample-size 480256 \
        --max-speech-positions 1876 \
        --required-batch-size-multiple 1 \
        --skip-invalid-size-inputs-valid-test \
        --keep-last-epochs 10 \
        --save-interval-updates 10000 \
        --disable-validation \
        --log-interval 10 \
        \
        --arch t5_transformer_base_asr \
        --share-input-output-embed \
        --find-unused-parameters \
        --bert-init \
        --relative-position-embedding \
        --mask-prob 0.0 \
        --mask-channel-prob 0.0 \
        \
        --finetune-from-model ${PT_CHECKPOINT_PATH}
```

#### Inference

Generating speech is available only if batch size is 1.

```
SPEECHT5_CODE_DIR=
CHECKPOINT_PATH=
DATA_ROOT=
SUBSET=
LABEL_DIR=
USER_DIR=
RESULTS_PATH=

python3 ${SPEECHT5_CODE_DIR}/SpeechT5/scripts/generate_speech.py ${DATA_ROOT} \
        --gen-subset test \
        --user-dir ${USER_DIR} \
        --task speecht5 \
        --t5-task s2s \
        --path ${CHECKPOINT_PATH} \
        --hubert-label-dir ${LABEL_DIR} \
        --batch-size 1 \
        --results-path ${RESULTS_PATH} \
        --sample-rate 16000
```

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) and [ESPnet](https://github.com/espnet/espnet) projects.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Reference

If you find our work is useful in your research, please cite the following paper:

```bibtex
@article{Ao2021SpeechT5,
  title   = {SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing},
  author  = {Junyi Ao and Rui Wang and Long Zhou and Chengyi Wang and Shuo Ren and Yu Wu and Shujie Liu and Tom Ko and Qing Li and Yu Zhang and Zhihua Wei and Yao Qian and Jinyu Li and Furu Wei},
  eprint={2110.07205},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2021}
}
```

### Contact Information

For help or issues using SpeechT5 models, please submit a GitHub issue.

For other communications related to SpeechT5, please contact Long Zhou (`lozhou@microsoft.com`).
