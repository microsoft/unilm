# TrOCR

## Introduction
TrOCR is an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, which leverages the Transformer architecture for both image understanding and wordpiece-level text generation. 

![TrOCR](https://pbs.twimg.com/media/FADdTXEVgAAsTWL?format=jpg&name=4096x4096)
 
 [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282), Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei, ```AAAI 2023```.

<!-- The TrOCR is currently implemented with the fairseq library. We hope to convert the models to the Huggingface format later. -->
The TrOCR models are also provided in the Huggingface format.[[Documentation](https://huggingface.co/docs/transformers/model_doc/trocr)][[Models](https://huggingface.co/models?filter=trocr)]

 
| Model                          |  #Param   | Test set | Score          |
|--------------------------------|-----------|----------|----------------|
| TrOCR-Small                    | 62M        | IAM     | 4.22 (Cased CER)     |
| TrOCR-Base                     | 334M       | IAM     | 3.42 (Cased CER)     |
| TrOCR-Large                    | 558M       | IAM     | 2.89 (Cased CER)     |
| TrOCR-Small                    | 62M        | SROIE   | 95.86 (F1)  |
| TrOCR-Base                     | 334M       | SROIE   | 96.34 (F1)  |
| TrOCR-Large                    | 558M       | SROIE   | 96.60 (F1)  |

| Model       | IIIT5K-3000 | SVT-647 | ICDAR2013-857 | ICDAR2013-1015 | ICDAR2015-1811 | ICDAR2015-2077 | SVTP-645 | CT80-288 |
|-------------|-------------|---------|---------------|----------------|----------------|----------------|----------|----------|
| TrOCR-Base (Word Accuracy)  | 93.4        | 95.2    | 98.4          | 97.4           | 86.9           | 81.2           | 92.1     | 90.6     |
| TrOCR-Large (Word Accuracy) | 94.1        | 96.1    | 98.4          | 97.3           | 88.1           | 84.1           | 93.0     | 95.1     |
## Installation
~~~bash
conda create -n trocr python=3.7
conda activate trocr
git clone https://github.com/microsoft/unilm.git
cd unilm
cd trocr
pip install pybind11
pip install -r requirements.txt
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" 'git+https://github.com/NVIDIA/apex.git'
~~~

## Fine-tuning and evaluation
|   Model  | Download |
| -------- | -------- |
| TrOCR-Small-IAM    | [trocr-small-handwritten.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-handwritten.pt) |
| TrOCR-Base-IAM     | [trocr-base-handwritten.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt) |
| TrOCR-Large-IAM    | [trocr-large-handwritten.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-handwritten.pt) |
| TrOCR-Small-SROIE  | [trocr-small-printed.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-printed.pt) |
| TrOCR-Base-SROIE   | [trocr-base-printed.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-printed.pt) |
| TrOCR-Large-SROIE  | [trocr-large-printed.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-printed.pt) |
| TrOCR-Small-Stage1 | [trocr-small-stage1.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-stage1.pt) |
| TrOCR-Base-Stage1  | [trocr-base-stage1.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-stage1.pt) |
| TrOCR-Large-Stage1 | [trocr-large-stage1.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-stage1.pt) |
| TrOCR-Base-STR | [trocr-base-str.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-str.pt) |
| TrOCR-Large-STR | [trocr-large-str.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-str.pt) |

|   Test set  | Download |
| --------| -------- |
| IAM     | [IAM.tar.gz](https://layoutlm.blob.core.windows.net/trocr/dataset/IAM.tar.gz) |
| SROIE   | [SROIE_Task2_Original.tar.gz](https://layoutlm.blob.core.windows.net/trocr/dataset/SROIE_Task2_Original.tar.gz) |
| STR Benchmarks   | [STR_BENCHMARKS.zip](https://layoutlm.blob.core.windows.net/trocr/dataset/STR_BENCHMARKS.zip) |



### Fine-tuning on IAM
~~~bash
export MODEL_NAME=ft_iam
export SAVE_PATH=/path/to/save/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}
export DATA=/path/to/data
mkdir ${LOG_DIR}
export BSZ=8
export valid_BSZ=16

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
    $(which fairseq-train) \
    --data-type STR --user-dir ./ --task text_recognition --input-size 384 \
    --arch trocr_large \   # or trocr_base
    --seed 1111 --optimizer adam --lr 2e-05 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm \
    --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} \
    --tensorboard-logdir ${LOG_DIR} --max-epoch 300 --patience 20 --ddp-backend legacy_ddp \
    --num-workers 8 --preprocess DA2 --update-freq 1 \
    --bpe gpt2 --decoder-pretrained roberta2 \ # --bpe sentencepiece --sentencepiece-model ./unilm3-cased.model --decoder-pretrained unilm ## For small models
    --finetune-from-model /path/to/model --fp16 \
    ${DATA} 
~~~

### Evaluation on IAM
~~~bash
export DATA=/path/to/data
export MODEL=/path/to/model
export RESULT_PATH=/path/to/result
export BSZ=16

$(which fairseq-generate) \
        --data-type STR --user-dir ./ --task text_recognition --input-size 384 \
        --beam 10 --scoring cer2 --gen-subset test --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --preprocess DA2 \
        --bpe gpt2 --dict-path-or-url https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt \ # --bpe sentencepiece --sentencepiece-model ./unilm3-cased.model --dict-path-or-url https://layoutlm.blob.core.windows.net/trocr/dictionaries/unilm3.dict.txt ## For small models
        --fp16 \
        ${DATA}
~~~

### Fine-tuning on SROIE
~~~bash
export MODEL_NAME=ft_SROIE
export SAVE_PATH=/path/to/save/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}
export DATA=/path/to/data
mkdir ${LOG_DIR}
export BSZ=16
export valid_BSZ=16

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
    $(which fairseq-train) \
    --data-type SROIE --user-dir ./ --task text_recognition --input-size 384 \
    --arch trocr_large \   # or trocr_base
    --seed 1111 --optimizer adam --lr 5e-05 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 --warmup-updates 800 --weight-decay 0.0001 --log-format tqdm \
    --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} \
    --save-dir ${SAVE_PATH} --tensorboard-logdir ${LOG_DIR} --max-epoch 300 \
    --patience 10 --ddp-backend legacy_ddp --num-workers 10 --preprocess DA2 \
    --bpe gpt2 --decoder-pretrained roberta2 \ # --bpe sentencepiece --sentencepiece-model ./unilm3-cased.model --decoder-pretrained unilm ## For small models
    --update-freq 16 --finetune-from-model /path/to/model --fp16 \
    ${DATA}
~~~

### Evaluation on SROIE
~~~bash
export DATA=/path/to/data
export MODEL=/path/to/model
export RESULT_PATH=/path/to/result
export BSZ=16
$(which fairseq-generate) \
        --data-type SROIE --user-dir ./ --task text_recognition --input-size 384 \
        --beam 10 --nbest 1 --scoring sroie --gen-subset test \
        --batch-size ${BSZ} --path ${MODEL} --results-path ${RESULT_PATH} \
        --bpe gpt2 --dict-path-or-url https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt \ # --bpe sentencepiece --sentencepiece-model ./unilm3-cased.model --dict-path-or-url https://layoutlm.blob.core.windows.net/trocr/dictionaries/unilm3.dict.txt ## For small models
        --preprocess DA2 \
        --fp16 \
        ${DATA}
~~~

### Fine-tuning on STR Benchmarks
~~~bash
export MODEL_NAME=ft_str_benchmarks
export SAVE_PATH=/path/to/save/${MODEL_NAME}
export LOG_DIR=log_${MODEL_NAME}
export DATA=/path/to/data
mkdir ${LOG_DIR}
export BSZ=8
export valid_BSZ=16

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8  \
        $(which fairseq-train)  \
        --data-type Receipt53K  --user-dir ./  --task text_recognition --input-size 384 \
        --arch trocr_large  \ # or trocr_base
        --seed 1111  --optimizer adam  --lr 2e-05  \
        --lr-scheduler inverse_sqrt  --warmup-init-lr 1e-8  --warmup-updates 500  \
        --weight-decay 0.0001  --log-format tqdm  --log-interval 10 \
        --batch-size ${BSZ}  --batch-size-valid ${valid_BSZ}  --save-dir ${SAVE_PATH}  \
        --tensorboard-logdir ${LOG_DIR}  --max-epoch 500  --patience 20 \
        --preprocess RandAugment  --update-freq 1  --ddp-backend legacy_ddp \
        --num-workers 8  --finetune-from-model /path/to/model  \
        --bpe gpt2  --decoder-pretrained roberta2 \
        ${DATA} 
~~~

### Evaluation on STR Benchmarks
~~~bash
export DATA=/path/to/data
export MODEL=/path/to/model
export RESULT_PATH=/path/to/result
export BSZ=16
$(which fairseq-generate) \
        --data-type Receipt53K --user-dir ./ --task text_recognition \
        --input-size 384 --beam 10 --nbest 1 --scoring wpa \
        --gen-subset test --batch-size ${BSZ} --bpe gpt2 \
        --dict-path-or-url https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt \
        --path ${MODEL} --results-path ${RESULT_PATH} \
        --preprocess RandAugment \
        ${DATA}
~~~

Please convert the output file to zip format using "convert_to_sroie_format.py" and submit it on the [website](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=2) to get the  score.

## An Inference Example
Please see detials in [pic_inference.py](https://github.com/microsoft/unilm/blob/master/trocr/pic_inference.py).

## Citation
If you want to cite TrOCR in your research, please cite the following paper:
``` latex
@misc{li2021trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2021},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree. Portions of the source code are based on the [fairseq](https://github.com/pytorch/fairseq) project. [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information
For help or issues using TrOCR, please submit a GitHub issue.

For other communications related to TrOCR, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).
