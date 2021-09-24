# TrOCR

## Introduction
TrOCR is an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, which leverages the Transformer architecture for both image understanding and wordpiece-level text generation. 
 
 [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://layoutlm.blob.core.windows.net/trocr/dataset/IAM.tar.gz) Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei
 
 
| Model                          |  #Param   | testset | score          |
|--------------------------------|-----------|---------|----------------|
| TrOCR-Base                     | 334M       | IAM     | 3.42 (CER)     |
| TrOCR-Large                    | 558M       | IAM     | 2.89 (CER)     |
| TrOCR-Base                     | 334M       | SROIE   | 96.34 (SROIE)  |
| TrOCR-Large                    | 558M       | SROIE   | 96.58 (SROIE)  |

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
|   model  | download |
| -------- | -------- |
| TrOCR-Base-IAM     | [download](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt) |
| TrOCR-Large-IAM    | [download](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-handwritten.pt) |
| TrOCR-Base-SROIE   | [download](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-printed.pt) |
| TrOCR-Large-SROIE  | [download](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-printed.pt) |
| TrOCR-Base-Stage1  | [download](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-stage1.pt) |
| TrOCR-Large-Stage1 | [download](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-stage1.pt) |


|   testset  | download |
| --------| -------- |
| IAM     | [download](https://layoutlm.blob.core.windows.net/trocr/dataset/IAM.tar.gz) |
| SROIE   | [download](https://layoutlm.blob.core.windows.net/trocr/dataset/SROIE_Task2_Original.tar.gz) |



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
    --data-type STR --user-dir ./ --task text_recognition \
    --arch beit_large_decoder_large \   # or beit_base_decoder_large
    --seed 1111 --optimizer adam --lr 2e-05 --bpe gpt2 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 --warmup-updates 500 --weight-decay 0.0001 --log-format tqdm \
    --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} --save-dir ${SAVE_PATH} \
    --tensorboard-logdir ${LOG_DIR} --max-epoch 300 --patience 20 --ddp-backend legacy_ddp \
    --num-workers 8 --preprocess DA2 --decoder-pretrained roberta2 --update-freq 1 \
    --finetune-from-model /path/to/model --fp16 \
    ${DATA} \
~~~

### Evaluation on IAM
~~~bash
export DATA=/path/to/data
export MODEL=/path/to/model
export RESULT_PATH=/path/to/result
export BSZ=16

$(which fairseq-generate) \
        --data-type STR --user-dir ./ --task text_recognition --input-size 384 \
        --beam 10 --bpe gpt2 --scoring cer2 --gen-subset test --batch-size ${BSZ} \
        --path ${MODEL} --results-path ${RESULT_PATH} --text-recog-gen --preprocess DA2 \
        --decoder-pretrained roberta2 --fp16 \
        ${DATA}
~~~

### Fine-tuning on SROIE
For fine-tuning and inference on printed text, the "--decoder-pretrained" need to be set to "roberta" in large model and "roberta2" in base model.
This is a legacy issue caused by version iteration of the code during the experiment.

For handwritten text, set it to "roberta2" is ok.

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
    --data-type SROIE --user-dir ./ --task text_recognition \
    --arch beit_large_decoder_large \   # or beit_base_decoder_large
    --seed 1111 --optimizer adam --lr 5e-05 --bpe gpt2 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-8 --warmup-updates 800 --weight-decay 0.0001 --log-format tqdm \
    --log-interval 10 --batch-size ${BSZ} --batch-size-valid ${valid_BSZ} \
    --save-dir ${SAVE_PATH} --tensorboard-logdir ${LOG_DIR} --max-epoch 300 \
    --patience 10 --ddp-backend legacy_ddp --num-workers 10 --preprocess DA2 \
    --decoder-pretrained roberta --update-freq 16 --finetune-from-model /path/to/model --fp16 \
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
        --beam 10 --nbest 1 --bpe gpt2 --scoring sroie --gen-subset test \
        --batch-size ${BSZ} --path ${MODEL} --results-path ${RESULT_PATH} \
        --text-recog-gen --preprocess DA2 \
        --decoder-pretrained roberta \ # base:roberta2 large:roberta 
        --fp16 \
        ${DATA}
~~~
The score reported by "fairseq-generate" will be lower than which calculated by offical calculated tools (about 0.03).

So please convert the output file to zip format using "convert_to_sroie_format.py" and submit it on the [website](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=2) to get the true score.

## Citation
If you find TrOCR useful in your research, please cite the following paper:
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