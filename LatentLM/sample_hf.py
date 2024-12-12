import argparse
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from accelerate.utils import set_seed

from safetensors.torch import load_file 
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
        "--prediction_type",
        type=str,
        default="epsilon",
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use Exponential Moving Average for the final model weights.")  
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=250)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="cosine", help="The beta schedule to use for DDPM.") 
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
    parser.add_argument("--image_name", type=str, default="sample.png")
    args = parser.parse_args()
    return args

@torch.no_grad()
def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    # Create model:
    vae, input_size, latent_size, flatten_input = load_vae(args.vae, args.image_size)
        
    model = All_models[args.model](
        input_size=input_size,
        in_channels=latent_size,
        num_classes=args.num_classes,
        flatten_input=flatten_input,
    ).to(device).to(dtype)
    # Initialize the scheduler
    noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule, prediction_type=args.prediction_type)

    model.eval()
    vae.eval()
    # Potentially load in the weights and states from a previous save
    if args.checkpoint:
        other_state = torch.load(os.path.join(args.checkpoint, "other_state.pth"))
        scaling_factor = other_state["scaling_factor"]
        bias_factor = other_state["bias_factor"]
        print(f"Scaling factor: {scaling_factor}, Bias factor: {bias_factor}")
        if args.use_ema and other_state["ema"] is not None:
            checkpoint = other_state["ema"]["shadow_params"]
            for model_param, ema_param in zip(model.parameters(), checkpoint):
                model_param.data = ema_param.data.to(device).to(dtype)
            print(f"Loaded model from checkpoint {args.checkpoint}, EMA applied.")
        else:
            if os.path.exists(os.path.join(args.checkpoint, "model.safetensors")):
                checkpoint = load_file(os.path.join(args.checkpoint, "model.safetensors"))
            elif os.path.exists(os.path.join(args.checkpoint, "pytorch_model")):
                checkpoint = torch.load(os.path.join(args.checkpoint, "pytorch_model", "mp_rank_00_model_states.pt"))["module"]
            else:
                raise ValueError(f"Could not find model checkpoint in {args.checkpoint}.")
            
            model.load_state_dict(checkpoint)
            print(f"Loaded model from checkpoint {args.checkpoint}.")

    # Labels to condition the model with (feel free to change):
    class_labels = [281, 282, 283, 284, 285, 4, 7, 963]
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    def p_sample(model, image):
        noise_scheduler.set_timesteps(args.ddpm_num_inference_steps)
        for t in noise_scheduler.timesteps:
            model_output = model(image, t.repeat(image.shape[0]).to(image))
            image = noise_scheduler.step(model_output, t, image).prev_sample
        return image

    # Create sampling noise:
    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)
    # Setup classifier-free guidance:
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    # Sample images:
    samples = model.sample_with_cfg(y, args.cfg_scale, p_sample)
    images = vae.decode(samples / scaling_factor - bias_factor)

    # Save and display images:
    save_image(images, f"visuals/{args.image_name}", nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved image to visuals/{args.image_name}")


if __name__ == "__main__":
    args = parse_args()
    main(args)