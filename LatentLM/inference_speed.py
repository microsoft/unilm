import argparse
import json
import os
import sys
import time
import torch
from tqdm import tqdm
from accelerate.utils import set_seed

from tokenizer_models import AutoencoderKL, load_vae

from schedule.dpm_solver import DPMSolverMultistepScheduler
from models import All_models


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="A seed to use for the random number generator. Can be negative to not set a seed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Transformer-L",
        help="The config of the model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=None,
        help="The number of heads to use in the key/value attention in the model.",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/tmp/ILSVRC/Data/CLS-LOC/train",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--ref_stat_path",
        type=str,
        default="/mnt/unilm/hangbo/beit3/t2i/assets/fid_stats/imagenet_256_val.npz",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help=(
            "The image_size for input images, all the images in the train/validation dataset will be resized to this"
            " image_size"
        ),
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--steps_per_class", type=int, default=50, help="Number of steps per class."
    )
    parser.add_argument("--force_diffusion", action="store_true", help="Whether to force the use of diffusion models.")  
    parser.add_argument("--use_ema", action="store_true", help="Whether to use Exponential Moving Average for the final model weights.")  
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=250)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="cosine", help="The beta schedule to use for DDPM.")
    parser.add_argument("--prediction_type", type=str, default="epsilon", help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    args = parser.parse_args()
    return args

def suppress_output(rank):  
    """Suppress output for all processes except the one with rank 0."""  
    if rank != 0:  
        sys.stdout = open(os.devnull, 'w')  

@torch.no_grad()
def main(args):
    set_seed(args.seed)
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    prefix = "ema" if args.use_ema else "standard"
    exp_name = f"{prefix}_{args.steps_per_class}_{args.cfg_scale}_{args.ddpm_beta_schedule}_{args.ddpm_num_inference_steps}"
    print(f"Exp_name {exp_name}")
    vae, input_size, latent_size, flatten_input = load_vae(args.vae, args.image_size)
        
    vae.eval()
    # Potentially load in the weights and states from a previous save
    model = All_models[args.model](
        input_size=input_size,
        in_channels=latent_size,
        num_kv_heads=args.num_kv_heads,
        num_classes=args.num_classes,
        flatten_input=flatten_input,
    ).to(device).to(dtype)
    noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule, prediction_type=args.prediction_type)
    model.eval()

    def p_sample(model, image):
        noise_scheduler.set_timesteps(args.ddpm_num_inference_steps)
        for t in noise_scheduler.timesteps:
            model_output = model(image, t.repeat(image.shape[0]).to(image))
            image = noise_scheduler.step(model_output, t, image).prev_sample
        return image
    
    start = time.time()
    for _ in tqdm(range(5)):
        y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)
        y_null = torch.full_like(y, args.num_classes, device=device)
        y = torch.cat([y, y_null], 0)
        # Sample images:
        samples = model.sample_with_cfg(y, args.cfg_scale, p_sample)
    end = time.time()
    print(args.model, args.batch_size)
    print(f"Time taken: {end - start}, FPS: {5 * args.batch_size / (end - start)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)