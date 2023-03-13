# Fine-tuning BEiT-3 on VQAv2 (Visual Question Answering)


## Setup

1. [Setup environment](../README.md#setup).
2. Download COCO [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip), [2015 test images](http://images.cocodataset.org/zips/test2015.zip), annotations ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)), and questions ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip)), then organize the dataset as following structure:

```
/path/to/your_data/
  train2014/            
    COCO_train2014_000000000009.jpg                
    ...
  val2014/              
    COCO_val2014_000000000042.jpg
    ...  
  test2015/              
    COCO_test2015_000000000001.jpg
    ...         
  vqa/
    v2_OpenEnded_mscoco_train2014_questions.json
    v2_OpenEnded_mscoco_val2014_questions.json
    v2_OpenEnded_mscoco_test2015_questions.json
    v2_OpenEnded_mscoco_test-dev2015_questions.json
    v2_mscoco_train2014_annotations.json
    v2_mscoco_val2014_annotations.json
```

We then generate the index json files using the following command. [beit3.spm](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm) is the sentencepiece model used for tokenizing texts.
```
from datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/path/to/your_data",
    tokenizer=tokenizer,
    annotation_data_path="/path/to/your_data/vqa",
)
```


## Example: Fine-tuning BEiT-3 on VQAv2 (Visual Question Answering)

The BEiT-3 **base** model can be finetuned on VQAv2 using 8 V100-32GB:

```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 16 \
        --layer_decay 1.0 \
        --lr 3e-5 \
        --update_freq 1 \
        --randaug \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.01 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --task_head_lr_weight 20 \
        --opt_betas 0.9 0.98 \
        --enable_deepspeed
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*16 = 128`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models)
- `--enable_deepspeed`: optional. If you use apex, please enable deepspeed.


The BEiT-3 **large** model can be finetuned on VQAv2 using 8 V100-32GB:

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_large_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 16 \
        --layer_decay 1.0 \
        --lr 2e-5 \
        --update_freq 1 \
        --randaug \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.15 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.01 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --task_head_lr_weight 20 \
        --opt_betas 0.9 0.98 \
        --enable_deepspeed \
        --checkpoint_activations
```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*16 = 128`.
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretrained-models)
- `--enable_deepspeed`: optional. If you use apex, please enable deepspeed.
- `--checkpoint_activations`: using gradient checkpointing for saving GPU memory


## Example: Evaluate BEiT-3 Finetuned model on VQAv2 (Visual Question Answering)

- Get the prediction file of the fine-tuned BEiT3-base model on VQAv2 test with 8 V100-32GB:
```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 16 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_480_vqa.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_prediction \
        --eval \
        --dist_eval
```

- Get the prediction file of the fine-tuned BEiT3-large model on VQAv2 test with 8 V100-32GB:
```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_large_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 16 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_large_patch16_480_vqa.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_prediction \
        --eval \
        --dist_eval
```

Please then submit the prediction file in the `output_dir` to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/overview) to obtain the VQAv2 test-dev and test-std results.
