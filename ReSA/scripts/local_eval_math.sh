export TORCH_IND_SYM_NODE_NO_SYMPY=1
cd llm/
torchrun --nproc_per_node=1 --nnodes=1 --master_port=29388 eval.py \
    --limit 512 --batch_size 4 \
    --checkpoint_dir /path/to/DeepSeek-R1-Distill-Qwen-1.5B --downstream_task math \
    --save_feature resa_0.1_32 \
    --output_folder /path/to/result/ \
    --resa_rec_freq 32 \
    --resa_sparse_ratio 0.1