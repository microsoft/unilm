import argparse
import json
import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch_fidelity
from utils import center_crop_arr, safe_blob_write

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
    parser.add_argument("--train_data_dir", type=str, default="/tmp/ILSVRC/Data/CLS-LOC/train", help="A folder containing the training data.")  
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
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--steps_per_class", type=int, default=50, help="Number of steps per class."
    )
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

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    
class RefImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item = np.array(item[0])
        item = torch.from_numpy(item).permute(2, 0, 1)
        return item

@torch.no_grad()
def main(args):
    prefix = "ema" if args.use_ema else "standard"
    exp_name = f"{prefix}_{args.steps_per_class}_{args.cfg_scale}_{args.ddpm_beta_schedule}_{args.ddpm_num_inference_steps}"
    print(f"Exp_name {exp_name}")
    image_path = os.path.join(args.checkpoint, f"images_{exp_name}.npz")
    print(f"Computing fidelity metrics from {image_path}...")
    images = np.load(image_path)["arr_0"]
    images = torch.from_numpy(images).permute(0, 3, 1, 2)
    print(images.shape)
    dataset = ImageDataset(images)
    ref_dataset = ImageFolder(args.train_data_dir, transform=transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)))
    ref_dataset = RefImageDataset(ref_dataset)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dataset,
        input2=ref_dataset,
        batch_size=args.batch_size,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        save_cpu_ram=True,
        verbose=True,
    )
    print(metrics_dict)
    # metrics_dict = torch_fidelity.calculate_metrics(
    #     input1=dataset,
    #     input2=ref_dataset,
    #     batch_size=args.batch_size,
    #     cuda=True,
    #     prc=True,
    #     prc_batch_size=args.batch_size,
    #     save_cpu_ram=True,
    #     verbose=True,
    # )
    # print(metrics_dict)


if __name__ == "__main__":
    args = parse_args()
    main(args)