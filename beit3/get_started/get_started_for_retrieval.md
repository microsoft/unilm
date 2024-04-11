# Fine-tuning BEiT-3 on Image-text Retrieval

## COCO Retrieval Setup

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

We then generate the index json files using the following command. [beit3.spm](https://github.com/addf400/files/releases/download/beit3/beit3.spm) is the sentencepiece model used for tokenizing texts.
```
from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

RetrievalDataset.make_coco_dataset_index(
    data_path="/path/to/your_data",
    tokenizer=tokenizer,
)
```


## Flickr30k Retrieval Setup

1. [Setup environment](README.md#setup).
2. Sign [flickr images request form](https://forms.illinois.edu/sec/229675) and download [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip), then organize the dataset as following structure:

```
/path/to/your_data/
  flickr30k-images/            
    2923475135.jpg                
    ...      
  dataset_flickr30k.json
```

We then generate the index json files using the following command. [beit3.spm](https://github.com/addf400/files/releases/download/beit3/beit3.spm) is the sentencepiece model used for tokenizing texts.
```
from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

RetrievalDataset.make_flickr30k_dataset_index(
    data_path="/path/to/your_data",
    tokenizer=tokenizer,
    karpathy_path="/path/to/your_data",
)
```


## Example: Fine-tuning BEiT-3 on Retrieval

The BEiT-3 **base** model can be finetuned on retrieval tasks using 16 V100-32GB:

```bash       
python -m torch.distributed.launch --nproc_per_node=16 run_beit3_finetuning.py \
        --model beit3_base_patch16_384 \
        --input_size 384 \
        --task coco_retrieval \
        --batch_size 192 \
        --layer_decay 0.65 \
        --lr 2e-4 \
        --epochs 15 \
        --warmup_epochs 3 \
        --drop_path 0.2 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_itc_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --enable_deepspeed \
        --checkpoint_activations
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `192*16 = 3072`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models)
- `--task`: **coco_retrieval** for COCO retrieval, **flickr30k** for Flickr30k retrieval
- `--lr`: 2e-4 for COCO retrieval, 1e-4 for Flickr30k retrieval
- `--epochs`: 15 for COCO retrieval, 20 for Flickr30k retrieval
- `--warmup_epochs`: 3 for COCO retrieval, 5 for Flickr30k retrieval
- `--checkpoint_activations`: using gradient checkpointing for saving GPU memory


The BEiT-3 **large** model can be finetuned on retrieval tasks using 2x16 V100-32GB:

```bash
python -m torch.distributed.launch --nproc_per_node=16 --nnodes=2 --node_rank=$NODE_RANK \
       --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_beit3_finetuning.py \
        --model beit3_large_patch16_384 \
        --input_size 384 \
        --task coco_retrieval \
        --batch_size 96 \
        --layer_decay 0.85 \
        --lr 5e-5 \
        --epochs 15 \
        --warmup_epochs 3 \
        --drop_path 0.2 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_itc_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --enable_deepspeed \
        --checkpoint_activations
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `96*32 = 3072`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models)
- `--task`: **coco_retrieval** for COCO retrieval, **flickr30k** for Flickr30k retrieval
- `--epochs`: 15 for COCO retrieval, 20 for Flickr30k retrieval
- `--warmup_epochs`: 3 for COCO retrieval, 5 for Flickr30k retrieval
- `--checkpoint_activations`: using gradient checkpointing for saving GPU memory


## Example: Evaluate BEiT-3 Fine-tuned model on COCO Retrieval and Flickr30k Retrieval

- Get the results of our fine-tuned BEiT3-base model on retrieval tasks using a single GPU:
```bash       
python -m torch.distributed.launch --nproc_per_node=1 run_beit3_finetuning.py \
        --model beit3_base_patch16_384 \
        --input_size 384 \
        --task coco_retrieval \
        --batch_size 16 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_384_coco_retrieval.pth \
        --data_path /path/to/your_data \
        --eval \
        --dist_eval
```
- `--task`: **coco_retrieval** for COCO retrieval, **flickr30k** for Flickr30k retrieval
- `--finetune`: **beit3_base_patch16_384_coco_retrieval.pth** for COCO retrieval, **beit3_base_patch16_384_f30k_retrieval.pth** for Flickr30k retrieval

- Get the results of our fine-tuned BEiT3-large model on retrieval tasks using a single GPU:
```bash       
python -m torch.distributed.launch --nproc_per_node=1 run_beit3_finetuning.py \
        --model beit3_large_patch16_384 \
        --input_size 384 \
        --task coco_retrieval \
        --batch_size 16 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_384_coco_retrieval.pth \
        --data_path /path/to/your_data \
        --eval \
        --dist_eval
```
- `--task`: **coco_retrieval** for COCO retrieval, **flickr30k** for Flickr30k retrieval
- `--finetune`: **beit3_large_patch16_384_coco_retrieval.pth** for COCO retrieval, **beit3_large_patch16_384_f30k_retrieval.pth** for Flickr30k retrieval
