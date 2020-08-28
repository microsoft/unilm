python -m torch.distributed.launch --nproc_per_node=4 ./examples/seq_labeling/run_seq_labeling.py --data_dir ./examples/seq_labeling/data \
    --model_type layoutlm \
    --model_name_or_path /export/home/sf_gitrepo/form_pretrain/pretrained_model/layoutlm-large-uncased/layoutlm-large-uncased \
    --do_lower_case \
    --max_seq_length 512 \
    --do_eval \
    --num_train_epochs 100.0 \
    --logging_steps 10 \
    --save_steps -1 \
    --output_dir results/funsd_large \
    --labels examples/seq_labeling/data/labels.txt \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --fp16 \
    #--overwrite_output_dir
