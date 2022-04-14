# DiT for Image Classification

This folder contains the image classification running instructions on DiT for RVL-CDIP.

## Usage
### Data Preparation

**RVL-CDIP**

Download the "rvl-cdip.tar.gz" from this [link](https://www.cs.ryerson.ca/~aharley/rvl-cdip/) (~37GB). Then extract it to `PATH-to-rvlcdip`.

### Evaluation
Following commands provide example to evaluate the fine-tuned checkpoints.
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=47770  run_class_finetuning.py \
        --model beit_base_patch16_224          #beit_base_patch16_224 / beit_large_patch16_224
        --data_path "/path/to/rvlcdip"
        --eval_data_path "/path/to/rvlcdip"
        --enable_deepspeed
        --nb_classes 16
        --eval
        --data_set rvlcdip
        --finetune /path/to/model.pth
        --output_dir output_dir
        --log_dir output_dir/tf
        --batch_size 256
        --abs_pos_emb
        --disable_rel_pos_bias
```

### Training
Fine-tune DiT on RVL-CDIP:
```bash
exp_name=dit-base-exp

mkdir -p output/${exp_name}
python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py
        --model beit_base_patch16_224           #beit_base_patch16_224 / beit_large_patch16_224
        --data_path "/path/to/rvlcdip"
        --eval_data_path "/path/to/rvlcdip"
        --nb_classes 16
        --data_set rvlcdip
        --finetune /path/to/model.pth
        --output_dir output/${exp_name}/ 
        --log_dir output/${exp_name}/tf 
        --batch_size 64 
        --lr 5e-4 
        --update_freq 2 
        --eval_freq 10 
        --save_ckpt_freq 10 
        --warmup_epochs 20 
        --epochs 180 
        --layer_scale_init_value 1e-5 
        --layer_decay 0.75 
        --drop_path 0.2 
        --weight_decay 0.05 
        --clip_grad 1.0 
        --abs_pos_emb 
        --disable_rel_pos_bias 
```


## Citation

If you find this repository useful, please consider citing our work:
```
@misc{li2022dit,
    title={DiT: Self-supervised Pre-training for Document Image Transformer},
    author={Junlong Li and Yiheng Xu and Tengchao Lv and Lei Cui and Cha Zhang and Furu Wei},
    year={2022},
    eprint={2203.02378},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## Acknowledgment
This part is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, the [Beit](https://github.com/microsoft/unilm/tree/master/beit) repository, the [DeiT](https://github.com/facebookresearch/deit) repository and the [Dino](https://github.com/facebookresearch/dino) repository.
