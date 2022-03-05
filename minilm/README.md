# MiniLM
**Small and fast pre-trained models for language understanding and generation**

**\*\*\*\*\* New June 9, 2021: MiniLM v2 release \*\*\*\*\***

**MiniLM v2**: the pre-trained models for the paper entitled "[MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers](https://arxiv.org/abs/2012.15828)". We generalize deep self-attention distillation in MiniLMv1 by using self-attention relation distillation for task-agnostic compression of pre-trained Transformers. The proposed method eliminates the restriction on the number of student’s attention heads. Our monolingual and multilingual small models distilled from different base and large size teacher models achieve competitive performance. 

**[Multilingual] Pre-trained Models**

| Model                                                                        | Teacher Model | Speedup   | #Param    | XNLI (Acc) | MLQA (F1)    |
|------------------------------------------------------------------------------|---------------|-----------|-----------|------------|--------------|
| **[L12xH384 mMiniLMv2](https://1drv.ms/u/s!AjHn0yEmKG8qiyGH2uvettAtQOov)**   | XLMR-Large    | 2.7x      | 117M      | 72.9       | 64.9         |
| **[L6xH384 mMiniLMv2](https://1drv.ms/u/s!AjHn0yEmKG8qiyC4-L624EmV2i7z)**    | XLMR-Large    | 5.3x      | 107M      | 69.3       | 59.0         |

We compress XLMR-Large into 12-layer and 6-layer models with 384 hidden size and report the zero-shot performance on XNLI and MLQA test set. 

**[English] Pre-trained Models**

| Model                                                                        | Teacher Model         | Speedup   | #Param    | MNLI-m (Acc) | SQuAD 2.0 (F1) |
|------------------------------------------------------------------------------|-----------------------|-----------|-----------|--------------|----------------|
| **[L6xH768 MiniLMv2](https://1drv.ms/u/s!AjHn0yEmKG8qiyUqoRUc6P1t0mk0)**     | RoBERTa-Large         | 2.0x      | 81M       | 87.0         | 81.6           |
| **[L12xH384 MiniLMv2](https://1drv.ms/u/s!AjHn0yEmKG8qiyM5cFiv7ew6uOO1)**    | RoBERTa-Large         | 2.7x      | 41M       | 86.9         | 82.3           |
| **[L6xH384 MiniLMv2](https://1drv.ms/u/s!AjHn0yEmKG8qix6eX4PZbP2_N2MO)**     | RoBERTa-Large         | 5.3x      | 30M       | 84.4         | 76.4           |
| [L6xH768 MiniLMv2](https://1drv.ms/u/s!AjHn0yEmKG8qix8oZl0UtY-KnJtY)         | BERT-Large Uncased    | 2.0x      | 66M       | 85.0         | 77.7           |
| [L6xH384 MiniLMv2](https://1drv.ms/u/s!AjHn0yEmKG8qix0MfI2hDsmK20cY)         | BERT-Large Uncased    | 5.3x      | 22M       | 83.0         | 74.3           |
| [L6xH768 MiniLMv2](https://1drv.ms/u/s!AjHn0yEmKG8qiyLmvLxXOSgpTxxm)         | BERT-Base Uncased     | 2.0x      | 66M       | 84.2         | 76.3           |
| [L6xH384 MiniLMv2](https://1drv.ms/u/s!AjHn0yEmKG8qiyQSo9sRDP4t3_jT)         | BERT-Base Uncased     | 5.3x      | 22M       | 82.8         | 72.9           |

The table presents the dev results of different small models on MNLI and SQuAD 2.0.

**\*\*\*\*\* September, 2020: MiniLM was accepted by NeurIPS 2020 \*\*\*\*\***

**\*\*\*\*\* April 5, 2020: Multilingual MiniLM v1 release \*\*\*\*\***

**Multilingual MiniLM v1** (April 5, 2020): we released the **12**-layer multilingual MiniLM model with **384** hidden size distilled from [XLM-R Base](https://github.com/pytorch/fairseq/tree/master/examples/xlmr).

**\*\*\*\*\* February 29, 2020: MiniLM v1 release \*\*\*\*\***

**MiniLM v1** (February 29, 2020): the pre-trained models for the paper entitled "[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)". **Deep self-attention distillation is all you need** (for task-agnostic knowledge distillation of pre-trained Transformers). MiniLM (12-layer, 384-hidden) achieves 2.7x speedup and comparable results over BERT-Base (12-layer, 768-hidden) on **NLU** tasks as well as strong results on **NLG** tasks. The even smaller MiniLM (6-layer, 384-hidden) obtains 5.3x speedup and produces very competitive results.

# [Multilingual] Pretrained Model
The link to the pre-trained multilingual model:
- [Multilingual-MiniLMv1-L12-H384](https://1drv.ms/u/s!AjHn0yEmKG8qixM1DIyLGmdRqWYl?e=uq2kDh): 12-layer, 384-hidden, 12-heads, 21M Transformer parameters, 96M embedding parameters

## Usage
```python
>>> from transformers import AutoTokenizer, AutoModel
  
>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
>>> model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

Multilingual MiniLM uses the same tokenizer as XLM-R. But the Transformer architecture of our model is the same as BERT. We provide the fine-tuning code on XNLI based on [huggingface/transformers](https://github.com/huggingface/transformers). Please replace `run_xnli.py` in transformers with ours to fine-tune multilingual MiniLM.  

We evaluate the multilingual MiniLM on cross-lingual natural language inference benchmark (XNLI) and cross-lingual question answering benchmark (MLQA).

### Cross-Lingual Natural Language Inference - [XNLI](https://arxiv.org/abs/1809.05053)

We evaluate our model on cross-lingual transfer from English to other languages. Following [Conneau et al. (2019)](https://arxiv.org/abs/1911.02116), we select the best single model on the joint dev set of all the languages.

| Model                                                                                       | #Layers | #Hidden | #Transformer Parameters | Average | en   | fr   | es   | de   | el   | bg   | ru   | tr   | ar   | vi   | th   | zh   | hi   | sw   | ur   |
|---------------------------------------------------------------------------------------------|---------|---------|-------------------------|---------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| [mBERT](https://github.com/google-research/bert)                                            | 12      | 768     | 85M                     | 66.3    | 82.1 | 73.8 | 74.3 | 71.1 | 66.4 | 68.9 | 69.0 | 61.6 | 64.9 | 69.5 | 55.8 | 69.3 | 60.0 | 50.4 | 58.0 |
| [XLM-100](https://github.com/facebookresearch/XLM#pretrained-cross-lingual-language-models) | 16      | 1280    | 315M                    | 70.7    | 83.2 | 76.7 | 77.7 | 74.0 | 72.7 | 74.1 | 72.7 | 68.7 | 68.6 | 72.9 | 68.9 | 72.5 | 65.6 | 58.2 | 62.4 |
| [XLM-R Base](https://arxiv.org/abs/1911.02116)                                              | 12      | 768     | 85M                     | 74.5    | 84.6 | 78.4 | 78.9 | 76.8 | 75.9 | 77.3 | 75.4 | 73.2 | 71.5 | 75.4 | 72.5 | 74.9 | 71.1 | 65.2 | 66.5 |
| **mMiniLM-L12xH384**                                                                        | 12      | 384     | 21M                     | 71.1    | 81.5 | 74.8 | 75.7 | 72.9 | 73.0 | 74.5 | 71.3 | 69.7 | 68.8 | 72.1 | 67.8 | 70.0 | 66.2 | 63.3 | 64.2 |

This example code fine-tunes **12**-layer multilingual MiniLM on XNLI.

```bash
# run fine-tuning on XNLI
DATA_DIR=/{path_of_data}/
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_PATH=/{path_of_pre-trained_model}/

python ./examples/run_xnli.py --model_type minilm \
 --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR} \
 --model_name_or_path ${MODEL_PATH}/multilingual-minilm-l12-h384.bin --tokenizer_name xlm-roberta-base \
 --config_name ${MODEL_PATH}/multilingual-minilm-l12-h384-config.json --do_train --do_eval \
 --max_seq_length 128 --per_gpu_train_batch_size 128 \
 --learning_rate 5e-5 --num_train_epochs 5  --per_gpu_eval_batch_size 32 --weight_decay 0.001 \
 --warmup_steps 500 --save_steps 1500 --logging_steps 1500 --eval_all_checkpoints\
 --language en --fp16 --fp16_opt_level O2
```

### Cross-Lingual Question Answering - [MLQA](https://arxiv.org/abs/1910.07475)

Following [Lewis et al. (2019b)](https://arxiv.org/abs/1910.07475), we adopt SQuAD 1.1 as training data and use MLQA English development data for early stopping.

| Model F1 Score                                                                             | #Layers | #Hidden | #Transformer Parameters | Average | en   | es   | de   | ar   | hi   | vi   | zh   |
|--------------------------------------------------------------------------------------------|---------|---------|-------------------------|---------|------|------|------|------|------|------|------|
| [mBERT](https://github.com/google-research/bert)                                           | 12      | 768     | 85M                     | 57.7    | 77.7 | 64.3 | 57.9 | 45.7 | 43.8 | 57.1 | 57.5 |
| [XLM-15](https://github.com/facebookresearch/XLM#pretrained-cross-lingual-language-models) | 12      | 1024    | 151M                    | 61.6    | 74.9 | 68.0 | 62.2 | 54.8 | 48.8 | 61.4 | 61.1 |
| [XLM-R Base](https://arxiv.org/abs/1911.02116) (Reported)                                  | 12      | 768     | 85M                     | 62.9    | 77.8 | 67.2 | 60.8 | 53.0 | 57.9 | 63.1 | 60.2 |
| [XLM-R Base](https://arxiv.org/abs/1911.02116) (Our fine-tuned)                            | 12      | 768     | 85M                     | 64.9    | 80.3 | 67.0 | 62.7 | 55.0 | 60.4 | 66.5 | 62.3 |
| **mMiniLM-L12xH384**                                                                       | 12      | 384     | 21M                     | 63.2    | 79.4 | 66.1 | 61.2 | 54.9 | 58.5 | 63.1 | 59.0 |

# [English] Pre-trained Models
We release the **uncased** **12**-layer and **6**-layer MiniLM models with **384** hidden size distilled from an in-house pre-trained [UniLM v2](/unilm) model in BERT-Base size. We also release **uncased** **6**-layer MiniLM model with **768** hidden size distilled from [BERT-Base](https://github.com/google-research/bert). The models use the same WordPiece vocabulary as BERT.

The links to the pre-trained models:
- [MiniLMv1-L12-H384-uncased](https://1drv.ms/u/s!AjHn0yEmKG8qixAYyu2Fvq5ulnU7?e=DFApTA): 12-layer, 384-hidden, 12-heads, 33M parameters, 2.7x faster than BERT-Base
- MiniLMv1-L6-H384-uncased: 6-layer, 384-hidden, 12-heads, 22M parameters, 5.3x faster than BERT-Base
- MiniLMv1-L6-H768-uncased: 6-layer, 768-hidden, 12-heads, 66M parameters, 2.0x faster than BERT-Base

## Usage
```python
>>> from transformers import AutoTokenizer, AutoModel
  
>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
>>> model = AutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

## Fine-tuning on NLU tasks
MiniLM has the same Transformer architecture as BERT. For NLU tasks, our models in Pytorch version can be loaded using the BERT code in [huggingface/transformers](https://github.com/huggingface/transformers). The config file is needed to be replaced with MiniLM's.

We present the dev results on SQuAD 2.0 and several GLUE benchmark tasks.

| Model                                             | #Param | SQuAD 2.0 | MNLI-m | SST-2 | QNLI | CoLA | RTE  | MRPC | QQP  |
|---------------------------------------------------|--------|-----------|--------|-------|------|------|------|------|------|
| [BERT-Base](https://arxiv.org/pdf/1810.04805.pdf) | 109M   | 76.8      | 84.5   | 93.2  | 91.7 | 58.9 | 68.6 | 87.3 | 91.3 |
| **MiniLM-L12xH384**                               | 33M    | 81.7      | 85.7   | 93.0  | 91.5 | 58.5 | 73.3 | 89.5 | 91.3 |
| **MiniLM-L6xH384**                                | 22M    | 75.6      | 83.3   | 91.5  | 90.5 | 47.5 | 68.8 | 88.9 | 90.6 |

This example code fine-tunes **12**-layer MiniLM on SQuAD 2.0 dataset.

```bash
# run fine-tuning on SQuAD 2.0
DATA_DIR=/{path_of_data}/
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_PATH=/{path_of_pre-trained_model}/

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 ./examples/run_squad.py --model_type bert \
 --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR} \
 --model_name_or_path ${MODEL_PATH}/minilm-l12-h384-uncased.bin --tokenizer_name ${MODEL_PATH}/vocab.txt \
 --config_name ${MODEL_PATH}/minilm-l12-h384-uncased-config.json \
 --do_train --do_eval --do_lower_case \
 --train_file train-v2.0.json --predict_file dev-v2.0.json \
 --learning_rate 4e-5 --num_train_epochs 4 \
 --max_seq_length 384 --doc_stride 128 \
 --per_gpu_eval_batch_size=12 --per_gpu_train_batch_size=12 --save_steps 5000 \
 --version_2_with_negative
```

## Fine-tuning on NLG tasks
Following [UniLM](/unilm-v1), MiniLM can be fine-tuned as a sequence-to-sequence model by employing a specific self-attention mask to support various downstream NLG tasks. We use the [s2s-ft package](/s2s-ft) to conduct the fine-tuning for NLG tasks.

### Abstractive Summarization - [XSum](https://github.com/EdinburghNLP/XSum)

| Model                                                                                                      | #Param | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------------------------------------------------------------------------------------------------------|--------|---------|---------|---------|
| [BART (Lewis et al., 2019)](https://arxiv.org/pdf/1910.13461.pdf)                                          | 400M   | 45.14   | 22.27   | 37.25   |
| [MASS (Song et al., 2019)](https://github.com/microsoft/MASS#results-on-abstractive-summarization-9272019) | 123M   | 39.75   | 17.24   | 31.95   |
| [BertSumAbs (Liu and Lapata, 2019)](https://arxiv.org/pdf/1908.08345.pdf)                                  | 156M   | 38.76   | 16.33   | 31.15   |
| **MiniLM-L12xH384**                                                                                        | 33M    | 40.43   | 17.72   | 32.60   |
| **MiniLM-L6xH384**                                                                                         | 22M    | 38.79   | 16.39   | 31.10   |

This example code fine-tunes **12**-layer MiniLM on XSum dataset.

```bash
# run fine-tuning on XSum
TRAIN_FILE=/your/path/to/train.json
CACHED_FEATURE_FILE=/your/path/to/xsum_train.uncased.features.pt
OUTPUT_DIR=/your/path/to/save_checkpoints
CACHE_DIR=/your/path/to/transformer_package_cache
MODEL_PATH=/your/path/to/pre_trained_model/

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py \
  --train_file ${TRAIN_FILE} --cached_train_features_file ${CACHED_FEATURE_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --model_type bert --model_name_or_path ${MODEL_PATH}/minilm-l12-h384-uncased.bin \
  --tokenizer_name ${MODEL_PATH}/minilm-l12-h384-uncased-vocab-nlg.txt --config_name ${MODEL_PATH}/minilm-l12-h384-uncased-config.json \
  --do_lower_case --fp16 --fp16_opt_level O2 \
  --max_source_seq_length 464 --max_target_seq_length 48 \
  --per_gpu_train_batch_size 16 --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --num_warmup_steps 500 --num_training_steps 108000 --cache_dir ${CACHE_DIR}
```

```bash
# run decoding on XSum
MODEL_PATH=/your/path/to/model_checkpoint
VOCAB_PATH=/your/path/to/vocab_file
SPLIT=validation
INPUT_JSON=/your/path/to/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python decode_seq2seq.py \
  --fp16 --model_type bert --tokenizer_name ${VOCAB_PATH}/minilm-l12-h384-uncased-vocab-nlg.txt \
  --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 512 --max_tgt_length 48 --batch_size 32 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "." --need_score_traces
```

### Abstractive Summarization - [CNN / Daily Mail](https://github.com/harvardnlp/sent-summary)

| Model                                                                                                      | #Param | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------------------------------------------------------------------------------------------------------|--------|---------|---------|---------|
| [T5-11B (Raffel et al., 2019)](https://arxiv.org/pdf/1910.10683.pdf)                                       | 11B    | 43.52   | 21.55   | 40.69   |
| [BART (Lewis et al., 2019)](https://arxiv.org/pdf/1910.13461.pdf)                                          | 400M   | 44.16   | 21.28   | 40.90   |
| [UniLM V1 (Dong et al., 2019)](https://arxiv.org/abs/1905.03197)                                           | 340M   | 43.08   | 20.43   | 40.34   |
| [T5-Base (Raffel et al., 2019)](https://arxiv.org/pdf/1910.10683.pdf)                                      | 220M   | 42.05   | 20.34   | 39.40   |
| [MASS (Song et al., 2019)](https://github.com/microsoft/MASS#results-on-abstractive-summarization-9272019) | 123M   | 42.12   | 19.50   | 39.01   |
| [BertSumAbs (Liu and Lapata, 2019)](https://arxiv.org/pdf/1908.08345.pdf)                                  | 156M   | 41.72   | 19.39   | 38.76   |
| **MiniLM-L12*H384**                                                                                        | 33M    | 42.66   | 19.91   | 39.73   |
| **MiniLM-L6*H384**                                                                                         | 22M    | 41.57   | 19.21   | 38.64   |

### Question Generation - [SQuAD](https://arxiv.org/abs/1806.03822)

We present the results following the same [data split](https://github.com/xinyadu/nqg/tree/master/data) as in [(Du et al., 2017)](https://arxiv.org/pdf/1705.00106.pdf).

| Model                                                              | #Param | BLEU-4 | METEOR | ROUGE-L |
|--------------------------------------------------------------------|--------|--------|--------|---------|
| [(Du and Cardie, 2018)](https://www.aclweb.org/anthology/P18-1177) |        | 15.16  | 19.12  | -       |
| [(Zhang and Bansal, 2019)](https://arxiv.org/pdf/1909.06356.pdf)   |        | 18.37  | 22.65  | 46.68   |
| [UniLM V1 (Dong et al., 2019)](https://arxiv.org/abs/1905.03197)   | 340M   | 22.78  | 25.49  | 51.57   |
| **MiniLM-L12xH384**                                                | 33M    | 21.07  | 24.09  | 49.14   |
| **MiniLM-L6xH384**                                                 | 22M    | 20.31  | 23.43  | 48.21   |

We also report the results following the data split as in [(Zhao et al., 2018)](https://aclweb.org/anthology/D18-1424), which uses the reversed dev-test setup.

| Model                                                            | #Param | BLEU-4 | METEOR | ROUGE-L |
|------------------------------------------------------------------|--------|--------|--------|---------|
| [(Zhao et al., 2018)](https://aclweb.org/anthology/D18-1424)     |        | 16.38  | 20.25  | 44.48   |
| [(Zhang and Bansal, 2019)](https://arxiv.org/pdf/1909.06356.pdf) |        | 20.76  | 24.20  | 48.91   |
| [UniLM V1 (Dong et al., 2019)](https://arxiv.org/abs/1905.03197) | 340M   | 24.32  | 26.10  | 52.69   |
| **MiniLM-L12xH384**                                              | 33M    | 23.27  | 25.15  | 50.60   |
| **MiniLM-L6xH384**                                               | 22M    | 22.01  | 24.24  | 49.51   |

## Citation

If you find MiniLM useful in your research, please cite the following paper:

``` latex
@misc{wang2020minilm,
    title={MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers},
    author={Wenhui Wang and Furu Wei and Li Dong and Hangbo Bao and Nan Yang and Ming Zhou},
    year={2020},
    eprint={2002.10957},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using MiniLM, please submit a GitHub issue.

For other communications related to MiniLM, please contact Wenhui Wang (`wenwan@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).
