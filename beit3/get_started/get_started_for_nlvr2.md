# Fine-tuning BEiT-3 on NLVR2 (Visual Reasoning)


## Setup

1. [Setup environment](../README.md#setup).
2. Clone the [repository](https://github.com/lil-lab/nlvr) and sign the [request form](https://goo.gl/forms/yS29stWnFWzrDBFH3) to download the images, then organize the dataset as following structure:

```
/path/to/your_data/
  images/train/                            
    0/train-11670-0-img0.png
    ...
  dev/              
    dev-269-0-img0.png
    ...  
  test1/              
    test1-261-0-img0.png
    ...         
  nlvr/ (nlvr repo)
    nlvr/
    nlvr2/
```

We then generate the index json files using the following command. [beit3.spm](https://github.com/addf400/files/releases/download/beit3/beit3.spm) is the sentencepiece model used for tokenizing texts.
```
from datasets import NLVR2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

NLVR2Dataset.make_dataset_index(
    data_path="/path/to/your_data", 
    tokenizer=tokenizer, 
    nlvr_repo_path="/path/to/your_data/nlvr"
)
```


## Example: Fine-tuning BEiT-3 on NLVR2 (Visual Reasoning)

The BEiT-3 **base** model can be finetuned on NLVR2 using 8 V100-32GB:

```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_224 \
        --task nlvr2 \
        --batch_size 32 \
        --layer_decay 0.65 \
        --lr 7e-4 \
        --epochs 20 \
        --warmup_epochs 5 \
        --drop_path 0.2 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.2 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --enable_deepspeed
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*32 = 256`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models).
- `--enable_deepspeed`: optional. If you use apex, please enable deepspeed.
- `--lr`: 7e-4 for `BEiT3-base`, 5e-4 for `BEiT3-base-indomain`.


The BEiT-3 **large** model can be finetuned on NLVR2 using 8 V100-32GB:

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_large_patch16_224 \
        --task nlvr2 \
        --batch_size 32 \
        --layer_decay 0.85 \
        --lr 3e-4 \
        --epochs 20 \
        --warmup_epochs 5 \
        --drop_path 0.2 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.2 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --enable_deepspeed \
        --checkpoint_activations
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*32 = 256`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models).
- `--enable_deepspeed`: optional. If you use apex, please enable deepspeed.
- `--lr`: 3e-4 for `BEiT3-large`, 1e-4 for `BEiT3-large-indomain`.
- `--checkpoint_activations`: using gradient checkpointing for saving GPU memory.


## Example: Evaluate BEiT-3 Finetuned model on NLVR2 (Visual Reasoning)

- Get the result of our fine-tuned BEiT3-base model on NLVR2 test with 8 V100-32GB:
```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_224 \
        --task nlvr2 \
        --batch_size 32 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224_nlvr2.pth \
        --data_path /path/to/your_data \
        --eval \
        --dist_eval
```

Expected results:
```
* Acc 84.386
```

- Get the result of our fine-tuned BEiT3-large model on NLVR2 test with 8 V100-32GB:
```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_large_patch16_224 \
        --task nlvr2 \
        --batch_size 32 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_224_nlvr2.pth \
        --data_path /path/to/your_data \
        --eval \
        --dist_eval
```

Expected results:
```
* Acc 89.437
```
