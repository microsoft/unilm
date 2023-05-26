CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode="text-to-image" \
  --resume_from_checkpoint="textdiffuser-ckpt/diffusion_backbone" \
  --prompt="A sign that says 'Hello'" \
  --output_dir="./output" \
  --vis_num=4