# UniLM
**Unified pre-training for natural language understanding (NLU) and generation (NLG)**


**```Update: ```** **Check out the latest information and models of UniLM at [https://github.com/microsoft/unilm/tree/master/unilm](https://github.com/microsoft/unilm/tree/master/unilm)**


**\*\*\*\*\* October 1st, 2019: UniLM v1 release \*\*\*\*\***

**UniLM v1** (September 30th, 2019): the code and pre-trained models for the NeurIPS 2019 paper entitled "[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)". UniLM (v1) achieves the **new SOTA results** in **NLG** (especially **sequence-to-sequence generation**) tasks/benchmarks, including abstractive summarization (the Gigaword and CNN/DM dataset), question generation (the SQuAD QG dataset), etc. 

## Environment

### Docker

The recommended way to run the code is using docker under Linux:
```bash
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel bash
```

The docker is initialized by:
```bash
. .bashrc
apt-get update
apt-get install -y vim wget ssh

PWD_DIR=$(pwd)
cd $(mktemp -d)
git clone -q https://github.com/NVIDIA/apex.git
cd apex
git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python setup.py install --user --cuda_ext --cpp_ext
cd $PWD_DIR

pip install --user tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk
python -c "import nltk; nltk.download('punkt')"
pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval
```
The mixed-precision training code requires the specific version of [NVIDIA/apex](https://github.com/NVIDIA/apex/tree/1603407bf49c7fc3da74fceb6a6c7b47fece2ef8), which only supports pytorch<1.2.0.

Install the repo as a package in the docker:
```bash
mkdir ~/code; cd ~/code
git clone https://github.com/microsoft/unilm.git
cd ~/code/unilm/unilm-v1/src
pip install --user --editable .
```

## Pre-trained Models
We release both base-size and large-size **cased** UniLM models pre-trained with **Wikipedia and BookCorpus** corpora. The models are trained by using the same model configuration and WordPiece vocabulary as BERT. The model parameters can be loaded as in the fine-tuning code.

The links to the pre-trained models:
- [unilm1-large-cased](https://conversationhub.blob.core.windows.net/beit-share-public/ckpt/unilm1-large-cased.bin): 24-layer, 1024-hidden, 16-heads, 340M parameters
- [unilm1-base-cased](https://conversationhub.blob.core.windows.net/beit-share-public/ckpt/unilm1-base-cased.bin): 12-layer, 768-hidden, 12-heads, 110M parameters

## Fine-tuning
We provide instructions on how to fine-tune UniLM as a sequence-to-sequence model to support various downstream natural language generation tasks as follows. It is recommended to use 2 or 4 v100-32G GPU cards to fine-tune the model. Gradient accumulation (`--gradient_accumulation_steps`) can be enabled if there is an OOM error.

### Abstractive Summarization - [Gigaword](https://github.com/harvardnlp/sent-summary) (10K)

In the example, only 10K examples of the Gigaword training data are used to fine-tune UniLM. As shown in the following table, pre-training significantly improves performance for low-resource settings.

| Model                                                               | ROUGE-1   | ROUGE-2   | ROUGE-L   |
| ------------------------------------------------------------------- | --------- | --------- | --------- |
| [Transformer](http://proceedings.mlr.press/v97/song19d/song19d.pdf) | 10.97     | 2.23      | 10.42     |
| **UniLM**                                                           | **34.21** | **15.28** | **31.54** |

The data can be downloaded from [here](https://drive.google.com/open?id=1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV).

```bash
# run fine-tuning
DATA_DIR=/{path_of_data}/gigaword
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_RECOVER_PATH=/{path_of_pre-trained_model}/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biunilm/run_seq2seq.py --do_train --fp16 --amp --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} --src_file train.src.10k --tgt_file train.tgt.10k \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_position_embeddings 192 \
  --trunc_seg a --always_truncate_tail --max_len_b 64 \
  --mask_prob 0.7 --max_pred 64 \
  --train_batch_size 128 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 30
```

We provide a fine-tuned checkpoint (downloaded from [here](https://drive.google.com/open?id=1yKFBpT2dbN5d6WBjFlJqlXs9DQKCbRWe)) used for decoding. The inference and evaluation process is conducted as follows:
```bash
# run decoding
DATA_DIR=/{path_of_data}/gigaword
MODEL_RECOVER_PATH=/{path_of_fine-tuned_model}/ggw10k_model.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
# run decoding
python biunilm/decode_seq2seq.py --fp16 --amp --bert_model bert-large-cased --new_segment_ids --mode s2s --need_score_traces \
  --input_file ${DATA_DIR}/${EVAL_SPLIT}.src --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_tgt_length 32 \
  --batch_size 64 --beam_size 5 --length_penalty 0 \
  --forbid_duplicate_ngrams --forbid_ignore_word "."
# run evaluation
python gigaword/eval.py --pred ${MODEL_RECOVER_PATH}.${EVAL_SPLIT} \
  --gold ${DATA_DIR}/org_data/${EVAL_SPLIT}.tgt.txt --perl
```

The program `eval.py` generates a post-processed output file `${MODEL_RECOVER_PATH}.${EVAL_SPLIT}.post` (downloaded from [here](https://drive.google.com/open?id=15R7IvOVT3irdH3d2eqHPs_5Lbh1LIy8U)).

### Abstractive Summarization - [Gigaword](https://github.com/harvardnlp/sent-summary)

The training set of Gigaword contains 3.8M examples for headline generation.

| Model                                                                            | ROUGE-1   | ROUGE-2   | ROUGE-L   |
| -------------------------------------------------------------------------------- | --------- | --------- | --------- |
| [OpenNMT](https://aclweb.org/anthology/P18-1015)                                 | 36.73     | 17.86     | 33.68     |
| [Re3Sum (Cao et al., 2018)](https://aclweb.org/anthology/P18-1015)               | 37.04     | 19.03     | 34.46     |
| [MASS (Song et al., 2019)](http://proceedings.mlr.press/v97/song19d/song19d.pdf) | 38.73     | 19.71     | 35.96     |
| [BertShare (Rothe et al., 2019)](https://arxiv.org/pdf/1907.12461.pdf)           | 38.13     | 19.81     | 35.62     |
| **UniLM**                                                                        | **38.90** | **20.05** | **36.00** |

The data can be downloaded from [here](https://drive.google.com/open?id=1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV). **Note**: We use HarvardNLP's pre-processed version for comparisons with other models, which substitutes all the digits with the special token "#". If you would like to directly use the provided model, you may need additional postprocess to recover the correct digits.

```bash
# run fine-tuning
DATA_DIR=/{path_of_data}/gigaword
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_RECOVER_PATH=/{path_of_pre-trained_model}/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biunilm/run_seq2seq.py --do_train --fp16 --amp --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_position_embeddings 192 \
  --trunc_seg a --always_truncate_tail --max_len_a 0 --max_len_b 64 \
  --mask_prob 0.7 --max_pred 48 \
  --train_batch_size 128 --gradient_accumulation_steps 1 \
  --learning_rate 0.00003 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 30
```

The size of full training data (3.8M) is quite large. We can stop the fine-tuning procedure after 10 epochs.

We provide a fine-tuned checkpoint (downloaded from [here](https://drive.google.com/open?id=1jOI2nO16Uz4a0OWZ7Ro-jnD54MHMDlsv)) used for decoding. The inference and evaluation process is conducted as follows:
```bash
DATA_DIR=/{path_of_data}/gigaword
MODEL_RECOVER_PATH=/{path_of_fine-tuned_model}/ggw38m_model.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
# run decoding
python biunilm/decode_seq2seq.py --fp16 --amp --bert_model bert-large-cased --new_segment_ids --mode s2s --need_score_traces \
  --input_file ${DATA_DIR}/${EVAL_SPLIT}.src --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_tgt_length 32 \
  --batch_size 64 --beam_size 5 --length_penalty 0 \
  --forbid_duplicate_ngrams --forbid_ignore_word "."
# apply length penalty
python biunilm/gen_seq_from_trace.py --bert_model bert-large-cased --alpha 0.6 \
  --input ${MODEL_RECOVER_PATH}.${EVAL_SPLIT}
# run evaluation
python gigaword/eval.py --pred ${MODEL_RECOVER_PATH}.${EVAL_SPLIT}.alp0.6 \
  --gold ${DATA_DIR}/org_data/${EVAL_SPLIT}.tgt.txt --perl
```

The program `eval.py` generates a post-processed output file `${MODEL_RECOVER_PATH}.${EVAL_SPLIT}.alp0.6.post` (downloaded from [here](https://drive.google.com/open?id=1oycvzMC6ZoWZV7BOt5OlZ7q0SxM_0Zc9)).

### Abstractive Summarization - [CNN / Daily Mail](https://github.com/harvardnlp/sent-summary)

| Model                                                                                                                                                     | ROUGE-1   | ROUGE-2   | ROUGE-L   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | --------- | --------- |
| [PGNet (See et al., 2017)](https://www.aclweb.org/anthology/P17-1099)                                                                                     | 39.53     | 17.28     | 36.38     |
| [Bottom-Up (Gehrmann et al., 2018)](https://www.aclweb.org/anthology/D18-1443)                                                                            | 41.22     | 18.68     | 38.34     |
| [GPT-2 TL;DR: (Radford et al., 2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 29.34     | 8.27      | 26.58     |
| [MASS (Song et al., 2019)](https://github.com/microsoft/MASS#results-on-abstractive-summarization-9272019)                                                | 42.12     | 19.50     | 39.01     |
| [BertShare (Rothe et al., 2019)](https://arxiv.org/pdf/1907.12461.pdf)                                                                                    | 39.25     | 18.09     | 36.45     |
| [BertSumAbs (Liu and Lapata, 2019)](https://arxiv.org/pdf/1908.08345.pdf)                                                                                 | 41.72     | 19.39     | 38.76     |
| **UniLM**                                                                                                                                                 | **43.08** | **20.43** | **40.34** |

The data can be downloaded from [here](https://drive.google.com/open?id=1jiDbDbAsqy_5BM79SmX6aSu5DQVCAZq1).

```bash
# run fine-tuning
DATA_DIR=/{path_of_data}/cnn_dailymail
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_RECOVER_PATH=/{path_of_pre-trained_model}/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biunilm/run_seq2seq.py --do_train --fp16 --amp --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 768 --max_position_embeddings 768 \
  --trunc_seg a --always_truncate_tail \
  --max_len_a 568 --max_len_b 200 \
  --mask_prob 0.7 --max_pred 140 \
  --train_batch_size 48 --gradient_accumulation_steps 2 \
  --learning_rate 0.00003 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 30  
```

We provide a fine-tuned checkpoint (downloaded from [here](https://drive.google.com/open?id=1RyJxShxC9tDYVAyZwUwqkSoQ3l5DfjuE)) used for decoding. The inference and evaluation process is conducted as follows:

```bash
DATA_DIR=/{path_of_data}/cnn_dailymail
MODEL_RECOVER_PATH=/{path_of_fine-tuned_model}/cnndm_model.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
# run decoding
python biunilm/decode_seq2seq.py --fp16 --amp --bert_model bert-large-cased --new_segment_ids --mode s2s --need_score_traces \
  --input_file ${DATA_DIR}/${EVAL_SPLIT}.src --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 768 --max_tgt_length 128 \
  --batch_size 64 --beam_size 5 --length_penalty 0 \
  --forbid_duplicate_ngrams --forbid_ignore_word ".|[X_SEP]"
# apply length penalty
python biunilm/gen_seq_from_trace.py --bert_model bert-large-cased --alpha 1.0 \
  --input ${MODEL_RECOVER_PATH}.${EVAL_SPLIT}
# run evaluation
python cnndm/eval.py --pred ${MODEL_RECOVER_PATH}.${EVAL_SPLIT}.alp1.0 \
  --gold ${DATA_DIR}/org_data/${EVAL_SPLIT}.summary --trunc_len 70 --perl
```

The program `eval.py` generates a post-processed output file `${MODEL_RECOVER_PATH}.${EVAL_SPLIT}.alp1.0.post` (downloaded from [here](https://drive.google.com/open?id=1p93XD0wo3YvyxZnNYywujtnQoNCDiTF7)).

### Question Generation - [SQuAD](https://arxiv.org/abs/1806.03822)

We present the results following the same [data split](https://github.com/xinyadu/nqg/tree/master/data) and [evaluation scripts](https://github.com/xinyadu/nqg/tree/master/qgevalcap) as in [(Du et al., 2017)](https://arxiv.org/pdf/1705.00106.pdf).

| Model                                                              | BLEU-4    | METEOR    | ROUGE-L   |
| ------------------------------------------------------------------ | --------- | --------- | --------- |
| [(Du and Cardie, 2018)](https://www.aclweb.org/anthology/P18-1177) | 15.16     | 19.12     | -         |
| [(Zhang and Bansal, 2019)](https://arxiv.org/pdf/1909.06356.pdf)   | 18.37     | 22.65     | 46.68     |
| **UniLM**                                                          | **22.78** | **25.49** | **51.57** |

We also report the results following the data split as in [(Zhao et al., 2018)](https://aclweb.org/anthology/D18-1424), which uses the reversed dev-test setup.

| Model                                                            | BLEU-4    | METEOR    | ROUGE-L   |
| ---------------------------------------------------------------- | --------- | --------- | --------- |
| [(Zhao et al., 2018)](https://aclweb.org/anthology/D18-1424)     | 16.38     | 20.25     | 44.48     |
| [(Zhang and Bansal, 2019)](https://arxiv.org/pdf/1909.06356.pdf) | 20.76     | 24.20     | 48.91     |
| **UniLM**                                                        | **24.32** | **26.10** | **52.69** |

Note: If we directly use the tokenized references provided by [Du et al. (2017)](https://arxiv.org/pdf/1705.00106.pdf), the results are (22.17 BLEU-4 / 25.47 METEOR / 51.53 ROUGE-L) on the [raw data split](https://github.com/xinyadu/nqg/tree/master/data), and (23.69 BLEU-4 / 26.08 METEOR / 52.70 ROUGE-L) in the reversed dev-test setup.

Our processed data can be downloaded from [here](https://drive.google.com/open?id=11E3Ij-ctbRUTIQjueresZpoVzLMPlVUZ).

```bash
# run fine-tuning
DATA_DIR=/{path_of_data}/qg/train
OUTPUT_DIR=/{path_of_fine-tuned_model}/
MODEL_RECOVER_PATH=/{path_of_pre-trained_model}/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biunilm/run_seq2seq.py --do_train --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} --src_file train.pa.tok.txt --tgt_file train.q.tok.txt \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_position_embeddings 512 \
  --mask_prob 0.7 --max_pred 48 \
  --train_batch_size 32 --gradient_accumulation_steps 2 \
  --learning_rate 0.00002 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 10
```

We provide a fine-tuned checkpoint (downloaded from [here](https://drive.google.com/open?id=1JN2wnkSRotwUnJ_Z-AbWwoPdP53Gcfsn)) used for decoding. The inference and evaluation process is conducted as follows:

```bash
DATA_DIR=/{path_of_data}/qg/test
MODEL_RECOVER_PATH=/{path_of_fine-tuned_model}/qg_model.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
# run decoding
python biunilm/decode_seq2seq.py --bert_model bert-large-cased --new_segment_ids --mode s2s \
  --input_file ${DATA_DIR}/$test.pa.tok.txt --split ${EVAL_SPLIT} --tokenized_input \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_tgt_length 48 \
  --batch_size 16 --beam_size 1 --length_penalty 0
# run evaluation using our tokenized data as reference
python qg/eval_on_unilm_tokenized_ref.py --out_file qg/output/qg.test.output.txt
# run evaluation using tokenized data of Du et al. (2017) as reference
python qg/eval.py --out_file qg/output/qg.test.output.txt
```

The files `qg/eval_on_unilm_tokenized_ref.py` and `qg/eval.py` are in Python 2.\*, because they are dependent on the [evaluation scripts](https://github.com/xinyadu/nqg/tree/master/qgevalcap) of [Du et al., (2017)](https://arxiv.org/pdf/1705.00106.pdf). The output files can be downloaded from [here](https://drive.google.com/open?id=1MdaRftgl_HMqN7DLvYmw-zKkvOBZCP6U). Notice that our model predictions are cased, while the gold outputs provided by Du et al., (2017) are uncased. So the predicted results need to be converted to lowercase before computing the evaluation metrics.

## FAQ

- Install ROUGE-1.5.5
  - If we would like to use the Perl script of ROUGE, it can be installed by following [instruction-1](https://gist.github.com/donglixp/d7eea02d57ba2e099746f8463c2f6597) and [instruction-2](https://github.com/bheinzerling/pyrouge#installation). The ROUGE-1.5.5 package (written in Perl) can be downloaded from [here](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5). We can also use the Python-version evaluation script by removing the flag `--perl` when running `eval.py`. Notice that there would be slight number difference between them due to the implementation details.
  
- [Run inference using CPUs](https://github.com/microsoft/unilm/issues/23#issuecomment-549788510)
  - Run `decode_seq2seq.py` without the flags `--amp` and `--fp16`, and uninstall the python package `nvidia/apex`.

## Citation

If you find UniLM useful in your work, you can cite the following paper:
```
@inproceedings{unilm,
    title={Unified Language Model Pre-training for Natural Language Understanding and Generation},
    author={Dong, Li and Yang, Nan and Wang, Wenhui and Wei, Furu and Liu, Xiaodong and Wang, Yu and Gao, Jianfeng and Zhou, Ming and Hon, Hsiao-Wuen},
    year={2019},
    booktitle = "33rd Conference on Neural Information Processing Systems (NeurIPS 2019)"
}
```

## Related Projects/Codebase

- Vision-Language Pre-training: https://github.com/LuoweiZhou/VLP
- MT-DNN: https://github.com/namisan/mt-dnn
- Response Generation Pre-training: https://github.com/microsoft/DialoGPT

## Acknowledgments
Our code is based on [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0). We thank the authors for their wonderful open-source efforts.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using UniLM, please submit a GitHub issue.

For other communications related to UniLM, please contact Li Dong (`lidong1@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).

