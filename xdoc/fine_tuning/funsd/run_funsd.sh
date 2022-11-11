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