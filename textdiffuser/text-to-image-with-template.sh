CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode="text-to-image-with-template" \
  --resume_from_checkpoint="textdiffuser-ckpt/diffusion_backbone" \
  --prompt="a poster of monkey music festival" \
  --template_image="assets/examples/text-to-image-with-template/case2.jpg" \
  --output_dir="./output" \
  --vis_num=4