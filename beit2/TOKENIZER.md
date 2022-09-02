# VQ-KD Training

The proposed VQ-KD aims at reconstructing semantic knowledge from the teacher rather than original pixels.
Then we can construct a highly compact semantic codebook for masked image modeling.

## Example: Training VQ-KD Tokenizer on ImageNet-1k

The VQ-KD model can be trained on ImageNet-1k using a DGX box (8 V100-32GB):

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_vqkd_training.py \
    --data_set image_folder \
    --data_path /path/to/imagenet-1k/train \
    --eval_data_path /path/to/imagenet-1k/eval \
    --output_dir /path/to/save/your_model \
    --log_dir /path/to/save/your_model \
    --process_type default \
    --train_interpolation bicubic \
    --min_crop_scale 0.08 \
    --model vqkd_encoder_base_decoder_3x768x12_clip \
    --teacher_input_size 224 \
    --codebook_n_emd 8192  \
    --codebook_emd_dim 32 \
    --quantize_kmeans_init \
    --rec_loss_type cosine \
    --batch_size 64 \
    --opt adamw \
    --opt_betas 0.9 0.99 \
    --weight_decay 1e-4  \
    --warmup_epochs 10 \
    --epochs 100 \
    --save_ckpt_freq 20 
```
- `--model`: one can modify the encoder, decoder and teacher model in [modeling_vqkd.py](modeling_vqkd.py) according to personal demands. 

# Example: Encode images

One can compress the input image into quantized codes like this:
```bash
python test_get_code.py
```

## Model Zoo
We provide some trained vq-kd tokenizers here.

| model name | encoder layers | decoder layers | teacher model | codebook usage | weight |
|------------|:--------------:|:--------------:|:-------------:|:--------------:|:-------:|
| vqkd_encoder_base_decoder_1x768x12_clip | 12 | 1 | CLIP ViT-B/16 | 100% | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_1x768x12_clip-d93179da.pth) |
| vqkd_encoder_base_decoder_3x768x12_clip | 12 | 3 | CLIP ViT-B/16 | 97% | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth) |
| vqkd_encoder_base_decoder_1x768x12_dino | 12 | 1 | DINO ViT-B/16 | 100% | [link](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_1x768x12_dino-663c55d7.pth) |
