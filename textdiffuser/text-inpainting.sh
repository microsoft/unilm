CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode="text-inpainting" \
  --resume_from_checkpoint="textdiffuser-ckpt/diffusion_backbone" \
  --prompt="a boy draws good morning on a board" \
  --original_image="assets/examples/text-inpainting/case2.jpg" \
  --text_mask="assets/examples/text-inpainting/case2_mask.jpg" \
  --output_dir="./output" \
  --vis_num=4