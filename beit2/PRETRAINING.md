# BEiT v2 Pretraining

We follow the settings proposed in [BEiT v1](https://github.com/microsoft/unilm/tree/master/beit).


## Example: Pre-training BEiT v2 on ImageNet-1k

### Base-size

The BEiT v2 **base** model can be pretrained on ImageNet-1k using a DGX-2 box (16 V100-32GB):

```bash
python -m torch.distributed.launch --nproc_per_node=16 run_beitv2_pretraining.py \
        --data_set image_folder \
        --data_path /path/to/imagenet-1k/train \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model \
        --model beit_base_patch16_224_8k_vocab_cls_pt \
        --shared_lm_head True \
        --early_layers 9 \
        --head_layers 2 \
        --num_mask_patches 75 \
        --second_input_size 224 \
        --second_interpolation bicubic \
        --min_crop_scale 0.2 \
        --tokenizer_model vqkd_encoder_base_decoder_3x768x12_clip \
        --tokenizer_weight https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth \
        --batch_size 128 \
        --lr 1.5e-3 \
        --warmup_epochs 10 \
        --clip_grad 3.0 \
        --drop_path 0.1 \
        --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std \
        --opt_betas 0.9 0.999 \
        --opt_eps 1e-8  \
        --epochs 1600 \
        --save_ckpt_freq 20 
```
- `--model`: beit v2 model. `beit_base_patch16_224_8k_vocab_cls_pt` means base model with cls-token pretraining. `beit_base_patch16_224_8k_vocab` means base model without cls-token pretraining, i.e., beit v1 base model.
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size`. So in the above example, the effective batch size is `128*16 = 2048`.
- `--tokenizer_model`: we recommand `vqkd_encoder_base_decoder_3x768x12_clip`.
- `--tokenizer_weight`: weight path of tokenizer model.
- `--epochs`: we use 300 for short schedules and 1600 for long schedules.
- `--opt_betas`: 0.98 for 300 epochs and 0.999 for 1600 epochs.
- `--drop_path`: 0. for 300 epochs and 0.1 for 1600 epochs.

### Large-size

The BEiT v2 **large** model can be pretrained on ImageNet-1k using 4xDGX-2 box (4x16 V100-32GB):

```bash
python -m torch.distributed.launch --nnodes 4 --node_rank {0, 1, 2, 3} --nproc_per_node=16 run_beitv2_pretraining.py \
        --data_set image_folder \
        --data_path /path/to/imagenet-1k/train \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model \
        --model beit_large_patch16_224_8k_vocab_cls_pt \
        --shared_lm_head True \
        --early_layers 21 \
        --head_layers 2 \
        --num_mask_patches 75 \
        --second_input_size 224 \
        --second_interpolation bicubic \
        --min_crop_scale 0.2 \
        --tokenizer_model vqkd_encoder_base_decoder_3x768x12_clip \
        --tokenizer_weight https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth \
        --batch_size 32 \
        --lr 1.5e-3 \
        --warmup_epochs 10 \
        --clip_grad 3.0 \
        --drop_path 0.1 \
        --layer_scale_init_value 1e-5 \
        --imagenet_default_mean_and_std \
        --opt_betas 0.9 0.999 \
        --opt_eps 1e-8  \
        --epochs 1600 \
        --save_ckpt_freq 20 
```
- `--model`: beit v2 model. `beit_large_patch16_224_8k_vocab_cls_pt` means large model with cls-token pretraining. `beit_large_patch16_224_8k_vocab` means large model without cls-token pretraining, i.e., beit v1 large model.
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size`. So in the above example, the effective batch size is `32*4x16 = 2048`.
- `--tokenizer_model`: we recommand `vqkd_encoder_base_decoder_3x768x12_clip`.
- `--tokenizer_weight`: weight path of tokenizer model.
- `--epochs`: we use 300 for short schedules and 1600 for long schedules.
- `--opt_betas`: 0.98 for 300 epochs and 0.999 for 1600 epochs.
- `--drop_path`: 0. for 300 epochs and 0.1 for 1600 epochs.

## Model Zoo

We provide some pretrained models here.

| model name | pretraining epochs | weight |
|------------|:------------------:|:------:|
| beit_base_patch16_224_8k_vocab_cls_pt | 300 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_300e.pth) |
| beit_base_patch16_224_8k_vocab_cls_pt | 1600 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth) |
| beit_large_patch16_224_8k_vocab_cls_pt | 300 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_300e.pth) |
| beit_large_patch16_224_8k_vocab_cls_pt | 1600 | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth) |
