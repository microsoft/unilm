import argparse
import json
import os
import sys
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
from accelerate.utils import set_seed

from safetensors.torch import load_file 
from tokenizer_models import AutoencoderKL, load_vae

from schedule.dpm_solver import DPMSolverMultistepScheduler
from models import All_models
from utils import safe_blob_dump
from metrics import compute_fid_without_store, compute_inception_score_from_tensor


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
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
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
    dist.init_process_group(backend="gloo", init_method='env://')
    rank = dist.get_rank()
    suppress_output(rank)
    print(args)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
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
    other_state = torch.load(os.path.join(args.checkpoint, "other_state.pth"), map_location="cpu")
    scaling_factor = other_state["scaling_factor"]
    bias_factor = other_state["bias_factor"]
    print(f"Scaling factor: {scaling_factor}, Bias factor: {bias_factor}")
    # Potentially load in the weights and states from a previous save
    latent_path = os.path.join(args.checkpoint, f"latent_{exp_name}.pth")
    if os.path.exists(latent_path) and not args.force_diffusion:
        all_latent_gather = torch.load(latent_path)
        print("Loaded latent from file.")
    else:
        model = All_models[args.model](
            input_size=input_size,
            in_channels=latent_size,
            num_classes=args.num_classes,
            flatten_input=flatten_input,
        ).to(device).to(dtype)
        noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule, prediction_type=args.prediction_type)
        model.eval()
        if args.checkpoint:
            if args.use_ema and other_state["ema"] is not None:
                checkpoint = other_state["ema"]["shadow_params"]
                for model_param, ema_param in zip(model.parameters(), checkpoint):
                    model_param.data = ema_param.data.to(device).to(dtype)
                print(f"Loaded model from checkpoint {args.checkpoint}, EMA applied.")
            else:
                if os.path.exists(os.path.join(args.checkpoint, "model.safetensors")):
                    checkpoint = load_file(os.path.join(args.checkpoint, "model.safetensors"))
                elif os.path.exists(os.path.join(args.checkpoint, "pytorch_model")):
                    checkpoint = torch.load(os.path.join(args.checkpoint, "pytorch_model", "mp_rank_00_model_states.pt"), map_location="cpu")["module"]
                
                model.load_state_dict(checkpoint)
                print(f"Loaded model from checkpoint {args.checkpoint}.")

        def p_sample(model, image):
            noise_scheduler.set_timesteps(args.ddpm_num_inference_steps)
            for t in noise_scheduler.timesteps:
                model_output = model(image, t.repeat(image.shape[0]).to(image))
                image = noise_scheduler.step(model_output, t, image).prev_sample
            return image

        all_latent = []
        class_start, class_end = args.num_classes // dist.get_world_size() * rank, args.num_classes // dist.get_world_size() * (rank + 1)
        classes = torch.arange(class_start, class_end, device=device).repeat(args.steps_per_class)
        classes = classes.chunk(math.ceil(classes.size(0) / args.batch_size))
        for y in tqdm(classes, disable=rank != 0):
            y_null = torch.full_like(y, args.num_classes, device=device)
            y = torch.cat([y, y_null], 0)
            # Sample images:
            samples = model.sample_with_cfg(y, args.cfg_scale, p_sample)
            all_latent.append(samples.float().cpu())

        all_latent = torch.cat(all_latent, 0)
        all_latent_gather = [torch.zeros_like(all_latent) for _ in range(dist.get_world_size())]
        dist.all_gather(all_latent_gather, all_latent)
        all_latent_gather = torch.cat(all_latent_gather, 0)
        if rank == 0:
            torch.save(all_latent_gather, latent_path)
        
    if rank == 0:
        all_images = torch.zeros((all_latent_gather.size(0), 3, 256, 256))
        if args.image_size != 256:
            transform = torch.nn.Upsample(size=(256, 256), mode="bilinear")
        else:
            transform = torch.nn.Identity()
        idx = 0
        for samples in tqdm(all_latent_gather.chunk(math.ceil(all_latent_gather.size(0) / args.batch_size))):
            images = vae.decode(samples.to(device).to(dtype) / scaling_factor - bias_factor)
            images = transform(images)
            images = (torch.clamp(images.float(), -1.0, 1.0) * 0.5 + 0.5).cpu().float()
            all_images[idx:idx + images.shape[0]] = images
            idx += images.shape[0]

        print(all_images.shape)
        fid_score = compute_fid_without_store(all_images, args.ref_stat_path, batch_size=args.batch_size, device=device)
        print(fid_score)
        IS_mean, IS_std = compute_inception_score_from_tensor(
            all_images, 
            batch_size=args.batch_size, 
            device=device, 
        )
        print(IS_mean, IS_std)
        result_path = os.path.join(args.checkpoint, f"result_{exp_name}.json")
        result = {
            "fid": fid_score.item(),
            "IS_mean": IS_mean.item(),
            "IS_std": IS_std.item(),
        }
        safe_blob_dump(result_path, result)
        image_path = os.path.join(args.checkpoint, f"images_{exp_name}.npz")
        all_images = (all_images * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        np.savez_compressed(image_path, all_images)


if __name__ == "__main__":
    args = parse_args()
    main(args)