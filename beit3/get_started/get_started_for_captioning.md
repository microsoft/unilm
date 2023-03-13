# Fine-tuning BEiT-3 on Image Captioning

## COCO Captioning Setup

1. [Setup environment](../README.md#setup).
2. Download [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip) and [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip), then organize the dataset as following structure:

```
/path/to/your_data/
  train2014/            
    COCO_train2014_000000000009.jpg                
    ...
  val2014/              
    COCO_val2014_000000000042.jpg
    ...       
  dataset_coco.json
```

We then generate the index json files using the following command. [beit3.spm](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm) is the sentencepiece model used for tokenizing texts.
```
from datasets import CaptioningDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

CaptioningDataset.make_coco_captioning_dataset_index(
    data_path="/path/to/your_data",
    tokenizer=tokenizer,
)
```


## NoCaps Setup

1. [Setup environment](README.md#setup).
2. Download [NoCaps val set](https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json), [NoCaps test set](https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json) and download imags using the urls in val and test json files, then organize the dataset as following structure:

```
/path/to/your_data/
  val/            
    09c863d76bcf6b00.jpg                
    ...
  test/              
    19dc6913830a0a21.jpg
    ...       
  nocaps_val_4500_captions.json
  nocaps_test_image_info.json
```

We then generate the index json files using the following command. [beit3.spm](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm) is the sentencepiece model used for tokenizing texts.
```
from datasets import CaptioningDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

CaptioningDataset.make_nocaps_captioning_dataset_index(
    data_path="/path/to/your_data",
)
```
We use COCO captioning training set as the training data of NoCaps.


## Example: Fine-tuning BEiT-3 on Captioning

The BEiT-3 **base** model can be fine-tuned on captioning tasks using 8 V100-32GB:

```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task coco_captioning \
        --batch_size 32 \
        --layer_decay 1.0 \
        --lr 4e-5 \
        --randaug \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --num_max_bpe_tokens 32 \
        --captioning_mask_prob 0.7 \
        --drop_worst_after 12000 \
        --dist_eval \
        --checkpoint_activations \
        --enable_deepspeed
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*32 = 256`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models).
- `--task`: **coco_captioning** for COCO captioning and **nocaps** for NoCaps dataset.
- `lr`: 4e-5 for COCO captioning and 1e-5 for NoCaps.
- `--enable_deepspeed`: optional. If you use apex, please enable deepspeed.
- `--checkpoint_activations`: using gradient checkpointing for saving GPU memory.


The BEiT-3 **large** model can be fine-tuned on captioning tasks using 8 V100-32GB:

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_large_patch16_480 \
        --input_size 480 \
        --task coco_captioning \
        --batch_size 32 \
        --layer_decay 1.0 \
        --lr 8e-6 \
        --randaug \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --num_max_bpe_tokens 32 \
        --captioning_mask_prob 0.7 \
        --drop_worst_after 12000 \
        --dist_eval \
        --checkpoint_activations \
        --enable_deepspeed
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*32 = 256`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models).
- `--task`: **coco_captioning** for COCO captioning and **nocaps** for NoCaps dataset.
- `lr`: 8e-6 for COCO captioning and NoCaps.
- `--enable_deepspeed`: optional. If you use apex, please enable deepspeed.
- `--checkpoint_activations`: using gradient checkpointing for saving GPU memory.


## Example: Evaluate BEiT-3 Fine-tuned model on Captioning

- Get the prediction file of the fine-tuned BEiT3-base model on captioning with 8 V100-32GB:
```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task coco_captioning \
        --batch_size 16 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_480_coco_captioning.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_prediction \
        --eval \
        --dist_eval
```
- `--task`: **coco_captioning** for COCO captioning and **nocaps** for NoCaps dataset.
- `--finetune`: **beit3_base_patch16_480_coco_captioning.pth** for COCO captioning and **beit3_base_patch16_480_nocaps.pth** for NoCaps dataset.

- Get the prediction file of the fine-tuned BEiT3-large model on captioning with 8 V100-32GB:
```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_large_patch16_480 \
        --input_size 480 \
        --task coco_captioning \
        --batch_size 16 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_480_coco_captioning.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_prediction \
        --eval \
        --dist_eval
```
- `--task`: **coco_captioning** for COCO captioning and **nocaps** for NoCaps dataset.
- `--finetune`: **beit3_large_patch16_480_coco_captioning.pth** for COCO captioning and **beit3_large_patch16_480_nocaps.pth** for NoCaps dataset.

Please then submit the prediction file in the `output_dir` to the [evaluation server](https://eval.ai/web/challenges/challenge-page/355/overview) to obtain the NoCaps val and test results.
