# FUNSD
The dataset will be **automatically downloaded**.

## Installation

```bash
pip install -r requirements.txt
```

Also, you need to install ```detectron2```. For example, if you use torch1.8 with cuda version 10.1, you can use the following command

```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 5678 run_funsd.py \
        --model_name_or_path /path/to/xdoc-pretrain-roberta-1M \
        --output_dir camera_ready_funsd_1M \
        --do_train \
        --do_eval \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16 \
        --overwrite_output_dir \
        --seed 42
```

## Test

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 5678 run_funsd.py \
        --model_name_or_path /path/to/xdoc-funsd \
        --output_dir camera_ready_funsd_1M \
        --do_eval \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16 \
        --overwrite_output_dir \
        --seed 42

```