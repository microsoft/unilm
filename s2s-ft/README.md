# s2s-ft: Sequence-to-Sequence Fine-Tuning
**A PyTorch package used to fine-tune pre-trained Transformers for sequence-to-sequence language generation**

## Environment

The recommended way to run the code is using docker:
```bash
docker run -it --rm --runtime=nvidia --ipc=host --privileged pytorch/pytorch:1.2-cuda10.0-cudnn7-devel bash
```

The following Python package need to be installed:
```bash
pip install --user methodtools py-rouge pyrouge nltk
python -c "import nltk; nltk.download('punkt')"
git clone https://github.com/NVIDIA/apex.git && cd apex && git reset --hard de6378f5dae8fcf2879a4be8ecea8bbcb9e59d5 && python setup.py install --cuda_ext --cpp_ext
```

Install the repo as a package:
```bash
git clone this repo into ${code_dir}

cd ${code_dir} ; pip install --editable .
```

## Pre-trained Models

We recommend to use the uncased model:
- [unilm1.2-base-uncased](https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased.bin): 12-layer, 768-hidden, 12-heads, 110M parameters
- [unilm2-base-uncased](https://unilm.blob.core.windows.net/ckpt/unilm2-base-uncased.bin): 12-layer, 768-hidden, 12-heads, 110M parameters

If you would like to use a cased model:
- [unilm1-base-cased](https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin): 12-layer, 768-hidden, 12-heads, 110M parameters
- [unilm1-large-cased](https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin): 24-layer, 1024-hidden, 16-heads, 340M parameters
- [unilm2-large-uncased](https://unilm.blob.core.windows.net/ckpt/unilm2-large-uncased.bin): 24-layer, 1024-hidden, 16-heads, 340M parameters
- [unilm2-large-cased](https://unilm.blob.core.windows.net/ckpt/unilm2-large-cased.bin): 24-layer, 1024-hidden, 16-heads, 340M parameters

If you prefer [small pretrained models](https://github.com/microsoft/unilm/tree/master/minilm) for faster inference speed:
- [minilm-l12-h384-uncased](https://1drv.ms/u/s!AjHn0yEmKG8qixAYyu2Fvq5ulnU7?e=DFApTA): 12-layer, 384-hidden, 12-heads, 33M parameters

## Input File Format

We support two dataset formats:

1. Text format: each line contains a json string of an example. `"src"` contains source sequence text, `"tgt"` contains target sequence text (`"tgt"` can be ignored for decoding). The data should be pre-processed as follows:

```bash
{"src": "Messages posted on social media claimed the user planned to `` kill as many people as possible ''", "tgt": "Threats to kill pupils in a shooting at a Blackpool school are being investigated by Lancashire police ."}
{"src": "Media playback is unsupported on your device", "tgt": "A slide running the entire length of one of the steepest city centre streets in Europe has been turned into a massive three-lane water adventure ."}
{"src": "Chris Erskine crossed low for Kris Doolan to tap home and give the Jags an early lead .", "tgt": "Partick Thistle will finish in the Scottish Premiership 's top six for the first time after beating Motherwell"}
```

2. Tokenized format: if you use tokenized data (with the same WordPiece tokenizers as BERT), `"src"` is a list of source sequence tokens, and `"tgt"` is a list of target sequence tokens (`"tgt"` can be ignored for decoding):

```bash
{"src": ["messages", "posted", "on", "social", "media", "claimed", "the", "user", "planned", "to", "\"", "kill", "as", "many", "people", "as", "possible", "\""], "tgt": ["threats", "to", "kill", "pupils", "in", "a", "shooting", "at", "a", "blackpool", "school", "are", "being", "investigated", "by", "lancashire", "police", "."]}
{"src": ["media", "playback", "is", "un", "##su", "##pp", "##orted", "on", "your", "device"], "tgt": ["a", "slide", "running", "the", "entire", "length", "of", "one", "of", "the", "steep", "##est", "city", "centre", "streets", "in", "europe", "has", "been", "turned", "into", "a", "massive", "three", "-", "lane", "water", "adventure", "."]}
{"src": ["chris", "erskine", "crossed", "low", "for", "kris", "doo", "##lan", "to", "tap", "home", "and", "give", "the", "ja", "##gs", "an", "early", "lead", "."], "tgt": ["part", "##ick", "thistle", "will", "finish", "in", "the", "scottish", "premiership", "'", "s", "top", "six", "for", "the", "first", "time", "after", "beating", "mother", "##well"]}
```

The code automatically detects the input format. If the json line contains `list`, we process the input as the tokenized format; if the json line contains `string`, the code will tokenize them.


## Example: [XSum](https://github.com/EdinburghNLP/XSum) with unilm1.2-base-uncased

### Fine-tuning

Pre-processed json dataset links: [text format](https://unilm.blob.core.windows.net/s2s-ft-data/xsum.json.zip), or [tokenized format](https://unilm.blob.core.windows.net/s2s-ft-data/xsum.uncased_tokenized.zip).

```bash
# path of training data
TRAIN_FILE=/your/path/to/train.json
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=/your/path/to/save_checkpoints
# folder used to cache package dependencies
CACHE_DIR=/your/path/to/transformer_package_cache

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
  --model_type unilm --model_name_or_path unilm1.2-base-uncased \
  --do_lower_case --fp16 --fp16_opt_level O2 --max_source_seq_length 464 --max_target_seq_length 48 \
  --per_gpu_train_batch_size 16 --gradient_accumulation_steps 1 \
  --learning_rate 7e-5 --num_warmup_steps 500 --num_training_steps 32000 --cache_dir ${CACHE_DIR}
```

- The fine-tuning batch size = `number of gpus` * `per_gpu_train_batch_size` * `gradient_accumulation_steps`. So in the above example, the batch size is `4*16*1 = 64`. The three arguments need to be adjusted together in order to remain the total batch size unchanged.
- `--do_lower_case`: for uncased models

### Decoding

```bash
# path of the fine-tuned checkpoint
MODEL_PATH=/your/path/to/model_checkpoint
SPLIT=validation
# input file that you would like to decode
INPUT_JSON=/your/path/to/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name unilm1.2-base-uncased --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 512 --max_tgt_length 48 --batch_size 32 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."
```

- The decoding results are saved at `${MODEL_PATH}.${SPLIT}`.
- `--do_lower_case`: for uncased models

### Evaluation

The golden answer text files can be downloaded at [here](https://unilm.blob.core.windows.net/s2s-ft-data/xsum.eval.zip).

```bash
SPLIT=validation
GOLD_PATH=/your/path/to/${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/eval_for_xsum.py --pred ${MODEL_PATH}.${SPLIT} --gold ${GOLD_PATH} --split ${SPLIT}
```


## Example: [XSum](https://github.com/EdinburghNLP/XSum) with minilm-l12-h384-uncased

### Fine-tuning

Pre-processed json dataset links: [text format](https://unilm.blob.core.windows.net/s2s-ft-data/xsum.json.zip), or [tokenized format](https://unilm.blob.core.windows.net/s2s-ft-data/xsum.uncased_tokenized.zip).

```bash
# path of training data
TRAIN_FILE=/your/path/to/train.json
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=/your/path/to/save_checkpoints
# folder used to cache package dependencies
CACHE_DIR=/your/path/to/transformer_package_cache

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
  --model_type minilm --model_name_or_path minilm-l12-h384-uncased \
  --do_lower_case --fp16 --fp16_opt_level O2 --max_source_seq_length 464 --max_target_seq_length 48 \
  --per_gpu_train_batch_size 16 --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --num_warmup_steps 500 --num_training_steps 108000 --cache_dir ${CACHE_DIR}
```

- The fine-tuning batch size = `number of gpus` * `per_gpu_train_batch_size` * `gradient_accumulation_steps`. So in the above example, the batch size is `4*16*1 = 64`. The three arguments need to be adjusted together in order to remain the total batch size unchanged.
- `--do_lower_case`: for uncased models

### Decoding

```bash
# path of the fine-tuned checkpoint
MODEL_PATH=/your/path/to/model_checkpoint
SPLIT=validation
# input file that you would like to decode
INPUT_JSON=/your/path/to/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_type minilm --tokenizer_name minilm-l12-h384-uncased --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 512 --max_tgt_length 48 --batch_size 32 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."
```

- The decoding results are saved at `${MODEL_PATH}.${SPLIT}`.
- `--do_lower_case`: for uncased models

### Evaluation

The golden answer text files can be downloaded at [here](https://unilm.blob.core.windows.net/s2s-ft-data/xsum.eval.zip).

```bash
SPLIT=validation
GOLD_PATH=/your/path/to/${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/eval_for_xsum.py --pred ${MODEL_PATH}.${SPLIT} --gold ${GOLD_PATH} --split ${SPLIT}
```


## Example: CNN / Daily Mail with unilm1-base-cased

Pre-processed json dataset links: [tokenized format](https://unilm.blob.core.windows.net/s2s-ft-data/cnndm.cased_tokenized.zip).

### Fine-tuning

```bash
# path of training data
export TRAIN_FILE=/your/path/to/train.json
# path used to cache training data
export CACHED_FEATURE_FILE=/your/path/to/cnndm_train.cased.features.pt
# folder used to save fine-tuned checkpoints
export OUTPUT_DIR=/your/path/to/save_checkpoints
# folder used to cache package dependencies
export CACHE_DIR=/your/path/to/transformer_package_cache

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py \
  --train_file $TRAIN_FILE --cached_train_features_file $CACHED_FEATURE_FILE --output_dir $OUTPUT_DIR \
  --model_type unilm --model_name_or_path unilm1-base-cased --fp16 --fp16_opt_level O2 \
  --max_source_seq_length 608 --max_target_seq_length 160 --per_gpu_train_batch_size 8 --gradient_accumulation_steps 2 \
  --learning_rate 7e-5 --num_warmup_steps 1000 --num_training_steps 45000 --cache_dir $CACHE_DIR --save_steps 1500
```

- The fine-tuning batch size = `number of gpus` * `per_gpu_train_batch_size` * `gradient_accumulation_steps`. So in the above example, the batch size is `4*8*2 = 64`. The three arguments need to be adjusted together in order to remain the total batch size unchanged.
- A fine-tuned checkpoint is provided at [here](https://unilm.blob.core.windows.net/ckpt/cnndm.unilm1-base-cased.bin).


### Decoding

```bash
# path of the fine-tuned checkpoint
MODEL_PATH=/your/path/to/model_checkpoint
SPLIT=dev
# input file that you would like to decode
INPUT_JSON=/your/path/to/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name unilm1-base-cased --input_file ${INPUT_JSON} --split $SPLIT \
  --model_path ${MODEL_PATH} --max_seq_length 768 --max_tgt_length 160 --batch_size 32 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."
```

- The decoding results are saved at `${MODEL_PATH}.${SPLIT}`.

### Evaluation

The golden answer text files can be downloaded at [here](https://unilm.blob.core.windows.net/s2s-ft-data/cnndm.eval.zip).

```bash
SPLIT=dev
GOLD_PATH=/your/path/to/${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/eval_for_cnndm.py --pred ${MODEL_PATH}.${SPLIT} --gold ${GOLD_PATH} --split ${SPLIT} --trunc_len 160
```




## Example: CNN / Daily Mail with unilm1.2-base-uncased

Pre-processed json dataset links: [tokenized format](https://unilm.blob.core.windows.net/s2s-ft-data/cnndm.uncased_tokenized.zip).

### Fine-tuning

```bash
# path of training data
export TRAIN_FILE=/your/path/to/train.json
# folder used to save fine-tuned checkpoints
export OUTPUT_DIR=/your/path/to/save_checkpoints
# folder used to cache package dependencies
export CACHE_DIR=/your/path/to/transformer_package_cache

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py \
  --train_file $TRAIN_FILE --output_dir $OUTPUT_DIR \
  --model_type unilm --model_name_or_path unilm1.2-base-uncased --do_lower_case --fp16 --fp16_opt_level O2 \
  --max_source_seq_length 608 --max_target_seq_length 160 --per_gpu_train_batch_size 8 --gradient_accumulation_steps 2 \
  --learning_rate 7e-5 --num_warmup_steps 1000 --num_training_steps 45000 --cache_dir $CACHE_DIR --save_steps 1500
```

- The fine-tuning batch size = `number of gpus` * `per_gpu_train_batch_size` * `gradient_accumulation_steps`. So in the above example, the batch size is `4*8*2 = 64`. The three arguments need to be adjusted together in order to remain the total batch size unchanged.
- `--do_lower_case`: for uncased models

### Decoding

```bash
# path of the fine-tuned checkpoint
MODEL_PATH=/your/path/to/model_checkpoint
SPLIT=dev
# input file that you would like to decode
INPUT_JSON=/your/path/to/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name unilm1.2-base-uncased --do_lower_case --input_file ${INPUT_JSON} --split $SPLIT \
  --model_path ${MODEL_PATH} --max_seq_length 768 --max_tgt_length 160 --batch_size 32 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "." --min_len 48
```

- The decoding results are saved at `${MODEL_PATH}.${SPLIT}`.

### Evaluation

The golden answer text files can be downloaded at [here](https://unilm.blob.core.windows.net/s2s-ft-data/cnndm.eval.zip).

```bash
SPLIT=dev
GOLD_PATH=/your/path/to/${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/eval_for_cnndm.py --pred ${MODEL_PATH}.${SPLIT} --gold ${GOLD_PATH} --split ${SPLIT} --trunc_len 160
```

## Citation

If you find this repository useful, please consider citing our work:
```
@article{s2s-ft,
      title={s2s-ft: Fine-Tuning Pretrained Transformer Encoders for Sequence-to-Sequence Learning}, 
      author={Hangbo Bao and Li Dong and Wenhui Wang and Nan Yang and Furu Wei},
      year={2021},
      eprint={2110.13640},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Example: XSum with unilm2-base-uncased

### Dataset

Pre-processed json dataset (tokenized format) [link](https://drive.google.com/file/d/1_Y2g-fGnAXjnFpMddp13Te-vfprX6iGH/view?usp=sharing)
### Fine-tuning

```bash
# choice the model to fine-tuning
export MODEL_NAME=unilm2-base-uncased
# path of training data
export TRAIN_FILE=/your/path/to/train.json
# folder used to save fine-tuned checkpoints
export OUTPUT_DIR=/your/path/to/save_checkpoints
# folder used to cache package dependencies
export CACHE_DIR=/your/path/to/transformer_package_cache
# learning rate
export LR=7e-5
# number of total training steps
export NUM_STEPS=48000
# target segment word drop prob
export TMP=0.4
# max length for source sequence
export MAX_SRC=720
# max length for target sequence
export MAX_TGT=48

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py \
  --train_file $TRAIN_FILE --output_dir $OUTPUT_DIR \
  --model_type unilm --model_name_or_path unilm2-base-uncased --do_lower_case \
  --fp16 --fp16_opt_level O2 \
  --max_source_seq_length $MAX_SRC --max_target_seq_length $MAX_TGT \
  --per_gpu_train_batch_size 8 --gradient_accumulation_steps 2 \
  --learning_rate $LR --num_warmup_steps 1000 --num_training_steps $NUM_STEPS \
  --cache_dir $CACHE_DIR --save_steps 1500 --target_mask_prob $TMP
```

- The fine-tuning batch size = `number of gpus` * `per_gpu_train_batch_size` * `gradient_accumulation_steps`. So in the above example, the batch size is `4*8*2 = 64`. The three arguments need to be adjusted together in order to remain the total batch size unchanged.
- `--do_lower_case`: for uncased models, no need for cased model
- `--target_mask_prob`: target segment word drop prob. For the xsum dataset, we recommend a value of 0.4 or 0.5. For the CNN/DailyMail dataset, we recommend a value of 0.7 for 0.8.
- `--learning_rate`: learning rate. For the base models, we recommend a value of 5e-5 ~ 1e-4. For the large models, we recommend a value of 1e-5 ~ 3e-5.

### Decoding

```bash
# tokenizer to decoding, same with the model name
export TOKENIZER_NAME=$MODEL_NAME  # or unilm2-base-uncased
# path of the fine-tuned checkpoint
export MODEL_PATH=/your/path/to/model_checkpoint
export SPLIT=dev
# input file that you would like to decode
export INPUT_JSON=/your/path/to/${SPLIT}.json
# max length for source sequence
export MAX_SRC=720
# max length for target sequence
export MAX_TGT=48

export MAX_LEN=$(($SRC_LEN+$TGT_LEN))
# set minimum length for decoding target sequence
export MIN_LEN=1

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_seq2seq.py \
  --fp16 --model_type unilm --tokenizer_name unilm2-base-uncased --do_lower_case \
  --input_file ${INPUT_JSON} --split $SPLIT --model_path ${MODEL_PATH} \
  --max_seq_length $MAX_LEN --max_tgt_length $MAX_TGT \
  --batch_size 24 --beam_size 8 \
  --length_penalty 0.9 --forbid_duplicate_ngrams --mode s2s \
  --forbid_ignore_word "." --min_len $MIN_LEN
```

- The decoding results are saved at `${MODEL_PATH}.${SPLIT}`.
- `--do_lower_case`: for uncased models, no need for cased model

### Evalation

The golden answer text files can be downloaded at [here](https://drive.google.com/file/d/142ANA2bJVYpg3bFfvda5zY7pj6Brg9RH/view?usp=sharing).

```bash
SPLIT=dev
GOLD_PATH=/your/path/to/${SPLIT}.target
# ${MODEL_PATH}.${SPLIT} is the predicted target file
python evaluations/evl_for_xsum.py --pred ${MODEL_PATH}.${SPLIT} --gold ${GOLD_PATH} --split ${SPLIT}
```

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
