# SQuAD
The dataset will be **automatically downloaded**.

## Installation
```
pip install -r requirements.txt
```

## Train
To train XDoc on SQuADv1.1

```bash
CUDA_VISIBLE_DEVICES=0 python run_squad.py \
  --model_name_or_path /path/to/xdoc-pretrain-roberta-1M \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./v1_result \
  --overwrite_output_dir
```

To train XDoc on SQuADv2.0

```bash
CUDA_VISIBLE_DEVICES=0 python run_squad.py \
  --model_name_or_path /path/to/xdoc-pretrain-roberta-1M \
  --dataset_name squad_v2 \
  --do_train \
  --do_eval \
  --version_2_with_negative \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./v2_result \
  --overwrite_output_dir
```

## Test
To test XDoc on SQuADv1.1


```bash
CUDA_VISIBLE_DEVICES=0 python run_squad.py \
  --model_name_or_path /path/to/xdoc-squad1.1 \
  --dataset_name squad \
  --do_eval \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./squadv1.1_result \
  --overwrite_output_dir
```

To test XDoc on SQuADv2.0

```bash
CUDA_VISIBLE_DEVICES=0 python run_squad.py \
  --model_name_or_path /path/to/xdoc-squad2.0 \
  --dataset_name squad_v2 \
  --do_eval \
  --version_2_with_negative \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./squadv2.0_result \
  --overwrite_output_dir
```

