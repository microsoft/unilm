# AdaLM

**Domain, language and task adaptation of pre-trained models.**

[Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains.](https://arxiv.org/abs/2106.13474)
Yunzhi Yao, Shaohan Huang, Wenhui Wang, Li Dong and Furu Wei, [ACL 2021](#)

This repository includes the code to finetune the adapted domain-specific model on downstrem tasks and the [code](https://github.com/microsoft/unilm/tree/master/adalm/incr_bpe) to generate incremental vocabulary for specific domain.

### Pre-trained Model
The adapted domain-specific model can be download:
- ***AdaLM-bio-base*** 12-layer, 768-hidden, 12-heads, 132M parameters || [One Drive](https://1drv.ms/u/s!AmcFNgkl1JIngxOqGWQk1u9G4mXf?e=Pa2RGC)
- ***AdaLM-bio-small*** 6-layer, 384-hidden, 12-heads, 34M parameters || [One Drive](https://1drv.ms/u/s!AmcFNgkl1JIngxQPKamwrRUelGUJ?e=qtmFHC)
- ***AdaLM-cs-base*** 12-layer, 768-hidden, 12-heads, 124M parameters || [One Drive](https://1drv.ms/u/s!AmcFNgkl1JIngxE_1VEP9gHU7mUe?e=XZemIz)
- ***AdaLM-cs-small*** 6-layer, 384-hidden, 12-heads, 30M parameters || [One Drive](https://1drv.ms/u/s!AmcFNgkl1JIngxJrUlHJbE4HY9Ev?e=PBaTNy)

### Fine-tuning Examples
#### Requirements

​	Install the requirements:

```bash
pip install -r requirements.txt
```

   Add the project to your PYTHONPATH

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`
```

#### Download Fine-tune Datasets

The biomedical downstream task can be download from [BLURB Leaderboard ](https://microsoft.github.io/BLURB/).  The computer science tasks can be download from  [allenai](https://github.com/allenai/dont-stop-pretraining)

#### Finetune Classification Task

```bash
# Set path to read training/dev dataset
export DATASET_PATH=/path/to/read/glue/task/data/            # Example: "/path/to/downloaded-glue-data-dir/mnli/"

# Set path to save the finetuned model and result score
export OUTPUT_PATH=/path/to/save/result_of_finetuning

export TASK_NAME=chemprot
# Set path to the model checkpoint you need to test 
export CKPT_PATH=/path/to/your/model/checkpoint

# Set config file
export CONFIG_FILE=/path/to/config/file

# Set vocab file
export VOCAB_FILE=/path/to/vocab/file

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export TRAIN_CACHE=${DATASET_PATH}/$TASK_NAME.bert.cache
export DEV_CACHE=${DATASET_PATH}/$TASK_NAME.bert.cache

# Setting the hyperparameters for the run.
export BSZ=32
export LR=1.5e-5
export EPOCH=30
export WD=0.1
export WM=0.1
CUDA_VISIBLE_DEVICES=0 python finetune/run_classifier.py \
   --model_type bert --model_name_or_path $CKPT_PATH \
   --config_name $CONFIG_FILE --tokenizer_name $VOCAB_FILE --do_lower_case\
   --data_dir $DATASET_PATH --cached_train_file $TRAIN_CACHE --cached_dev_file $DEV_CACHE \
   --do_train --do_eval --logging_steps 1000 --output_dir $OUTPUT_PATH --max_grad_norm 0 \
   --max_seq_length 128 --per_gpu_train_batch_size $BSZ --learning_rate $LR \
   --num_train_epochs $EPOCH --weight_decay $WD --warmup_ratio $WM \
   --fp16 --fp16_opt_level O2 --seed 42 --overwrite_output_dir

```



#### Finetune NER Task

To finetune the PICO task, just need to change the run_ner to run_pico.

```bash
# Set path to read training/dev dataset
export DATASET_PATH=/path/to/ner/task/data/           

# Set path to save the finetuned model and result score
export OUTPUT_PATH=/path/to/save/result_of_finetuning

export TASK_NAME=chemprot
# Set path to the model checkpoint you need to test 
export CKPT_PATH=/path/to/your/model/checkpoint

# Set config file
export CONFIG_FILE=/path/to/config/file

# Set vocab file
export VOCAB_FILE=/path/to/vocab/file

# Set label file  such as the BIO tag
export LABEL_FILE=/path/to/vocab/file

# Set path to cache train & dev features (tokenized, only use for this tokenizer!)
export CACHE_DIR=/path/to/cache

# Setting the hyperparameters for the run.
export BSZ=16
export LR=1.5e-5
export EPOCH=30
export WD=0.1
export WM=0.1
CUDA_VISIBLE_DEVICES=0 python finetune/run_ner.py \
   --model_type bert --model_name_or_path $CKPT_PATH \
   --config_name $CONFIG_FILE --tokenizer_name $VOCAB_FILE --do_lower_case\
   --data_dir $DATASET_PATH--cache_dir $CACHE_DIR --labels $LABEL_FILE \
   --do_train --do_eval --logging_steps 1000 --output_dir $OUTPUT_PATH --max_grad_norm 0 \
   --max_seq_length 128 --per_gpu_train_batch_size $BSZ --learning_rate $LR \
   --num_train_epochs $EPOCH --weight_decay $WD --warmup_ratio $WM \
   --fp16 --fp16_opt_level O2 --seed 42 --overwrite_output_dir
```

#### Results

**Biomedical**

|                 | JNLPBA    | PICO      | ChemProt  | Average   |
| --------------- | --------- | --------- | --------- | --------- |
| BERT            | 78.63     | 72.34     | 71.86     | 74.28     |
| BioBERT         | 79.35     | 73.18     | 76.14     | 76.22     |
| PubmedBERT      | **80.06** | 73.38     | 77.24     | 76.89     |
| AdaLM-bio-base  | 79.46     | **75.47** | **78.41** | **77.74** |
| AdaLM-bio-small | 79.04     | 74.91     | 72.06     | 75.34     |

**Computer Science**

|                | ACL-ARC   | SCIERC    | Average   |
| -------------- | --------- | --------- | --------- |
| BERT           | 64.92     | 81.14     | 73.03     |
| AdaLM-cs-base  | **73.61** | **81.91** | **77.76** |
| AdaLM-cs-small | 68.74     | 78.88     | 73.81     |

<!--

## Citation

If you find LayoutLM useful in your research, please cite the following paper:

``` latex
@misc{xu2019layoutlm,
    title={LayoutLM: Pre-training of Text and Layout for Document Image Understanding},
    author={Yiheng Xu and Minghao Li and Lei Cui and Shaohan Huang and Furu Wei and Ming Zhou},
    year={2019},
    eprint={1912.13318},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
-->

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using AdaLM, please submit a GitHub issue.

For other communications related to AdaLM, please contact Shaohan Huang (`shaohanh@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).



