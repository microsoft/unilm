# TextDiffuser: Diffusion Models as Text Painters (NeurIPS 2023)

<a href='https://arxiv.org/pdf/2305.10855.pdf'><img src='https://img.shields.io/badge/Arxiv-2305.10855-red'>
<a href='https://github.com/microsoft/unilm/tree/master/textdiffuser'><img src='https://img.shields.io/badge/Code-aka.ms/textdiffuser-yellow'>
<a href='https://jingyechen.github.io/textdiffuser/'><img src='https://img.shields.io/badge/Project Page-link-green'>
</a> [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TextDiffuser-blue)](https://huggingface.co/spaces/JingyeChen22/TextDiffuser)
<a href='https://colab.research.google.com/drive/115Qw0l5dhjlTtrbywMWRwhz9IxKE4_Dg?usp=sharing'><img src='https://img.shields.io/badge/GoogleColab-link-purple'>


TextDiffuser generates images with visually appealing text that is coherent with backgrounds. It is flexible and controllable to create high-quality text images using text prompts alone or together with text template images, and conduct text inpainting to reconstruct incomplete images with text.

<img src="assets/readme_images/introduction.jpg" width="80%">

## :star2:	Highlights

* We propose **TextDiffuser**, which is a two-stage diffusion-based framework for text rendering. It generates accurate and coherent text images from text prompts or additionally with template images, as well as conducting text inpainting to reconstruct incomplete images.

* We release **MARIO-10M**, containing large-scale image-text pairs with OCR annotations, including text recognition, detection, and character-level segmentation masks.

* We construct **MARIO-Eval**, a comprehensive text rendering benchmark containing 10k prompts.

* We **release the demo** at [link](https://huggingface.co/spaces/JingyeChen22/TextDiffuser). Welcome to use and provide feedbacks :hugs:.

## :stopwatch: News

- __[2023.09.22]__: :tada:	TextDiffuser is accepted to NeurIPS 2023.
- __[2023.06.22]__: Evaluation script is released.
- __[2023.06.15]__: :raised_hands:	:raised_hands:	:raised_hands: The Demo of TextDiffuser pre-trained with SD v2.1 is released in this [link](https://huggingface.co/spaces/JingyeChen22/TextDiffuser). Meanwhile, GoogleColab is available in this [link](https://colab.research.google.com/drive/115Qw0l5dhjlTtrbywMWRwhz9IxKE4_Dg?usp=sharing).
- __[2023.06.08]__: Training script is released.
- __[2023.06.07]__: MARIO-LAION is released.
- __[2023.06.02]__: :raised_hands:	:raised_hands:	:raised_hands:	Demo is available in this [link](https://huggingface.co/spaces/JingyeChen22/TextDiffuser).
- __[2023.05.26]__: Upload the inference code and checkpoint.
- __[2023.05.19]__: The paper is available at [link](https://arxiv.org/pdf/2305.10855.pdf).

## :hammer_and_wrench: Installation

Clone this repo: 
```
git clone github_path_to/TextDiffuser
cd TextDiffuser
```

Build up a new environment and install packages as follows:
```
conda create -n textdiffuser python=3.8
conda activate textdiffuser
pip install -r requirements.txt
```

Meanwhile, please install torch and torchvision that matches the version of system and cuda (refer to this [link](https://download.pytorch.org/whl/torch_stable.html)). 


Install Hugging Face Diffuser and replace some files:
```
git clone https://github.com/JingyeChen/diffusers
cp ./assets/files/scheduling_ddpm.py ./diffusers/src/diffusers/schedulers/scheduling_ddpm.py
cp ./assets/files/unet_2d_condition.py ./diffusers/src/diffusers/models/unet_2d_condition.py
cp ./assets/files/modeling_utils.py ./diffusers/src/diffusers/models/modeling_utils.py
cd diffusers && pip install -e .
```

Besides, a font file is needed for layout generation. Please put your font in ```assets/font/```. We recommend to use ```Arial.ttf```.



## :floppy_disk: Checkpoint

The checkpoints are in this [link](https://layoutlm.blob.core.windows.net/textdiffuser/textdiffuser-ckpt-new.zip?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) or [HFLink](https://huggingface.co/datasets/JingyeChen22/TextDiffuser/resolve/main/textdiffuser-ckpt-new.zip) (3.2GB). Please download it and unzip it. The file structures should be as follows:

 
 
```
textdiffuser
├── textdiffuser-ckpt
│   ├── diffusion_backbone/             # for diffusion backbone
│   ├── character_aware_loss_unet.pth   # for character-aware loss
│   ├── layout_transformer.pth          # for layout transformer
│   └── text_segmenter.pth              # for character-level segmenter
├── README.md
```

## :books: Dataset

<img src="assets/readme_images/laion-ocr.jpg" width="80%">


**MARIO-LAION**'s meta information is at this [link](https://layoutlm.blob.core.windows.net/textdiffuser/laion-ocr-new.zip?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) or [onedrive](https://mail2sysueducn-my.sharepoint.com/personal/huangyp28_mail2_sysu_edu_cn/_layouts/15/onedrive.aspx?ct=1686245253173&or=Teams%2DHL&ga=1&LOF=1&id=%2Fpersonal%2Fhuangyp28%5Fmail2%5Fsysu%5Fedu%5Fcn%2FDocuments%2Frelease%2Ftextdiffuser%2Fdata) (40GB), containing 9,194,613 samples. Please download it and unzip it by running ```python data/maion-laion-unzip.py```. The file structures of each folder should be as follows and ```data/maion-laion-example``` is provided for reference. We also provide ```data/visualize_charseg.ipynb``` to visualize the character-level segmentation mask.

```
├── 28330/
│   ├── 283305839/            
│   │   ├── caption.txt       # caption of the image
│   │   ├── charseg.npy       # character-level segmentation mask
│   │   ├── info.json         # more meta information given by laion, such as original height and width
├── ├── └── ocr.txt           # ocr detection and recognition results
```

The urls of each image is at this [link](https://layoutlm.blob.core.windows.net/textdiffuser/mario_laion_image_url.zip?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) or [onedrive](https://mail2sysueducn-my.sharepoint.com/personal/huangyp28_mail2_sysu_edu_cn/_layouts/15/onedrive.aspx?ct=1686245253173&or=Teams%2DHL&ga=1&LOF=1&id=%2Fpersonal%2Fhuangyp28%5Fmail2%5Fsysu%5Fedu%5Fcn%2FDocuments%2Frelease%2Ftextdiffuser%2Fdata) (794.6MB). The file structure is as follows:

```
├── maion_laion_image_url/
│   ├── mario-laion-url.txt         # urls for downloading by img2dataset
│   ├── mario-laion-index-url.txt   # urls and indices for each image
│   └── mario-laion-test-index.txt  # all indices for test dataset
```

Please download img2dataset wiht ```pip install img2dataset```, and download the images using the following command:
```
img2dataset --url_list=url.txt --output_folder=laion_ocr --thread_count=64  --resize_mode=no
```

After downloading, you need to resize each image to ```512x512```. Please follow ```mario-laion-index-url.txt``` to move each image to the corresponding folders. Images with indices in ```mario-laion-test-index.txt``` are used for testing. Please note that some links may be <span style="color:red">**invalid**</span>
 since the owners remove the images from their website.

## :steam_locomotive: Train
 
Please use ```accelerate config``` to configure your acceleration policy at first, then modify output_dir, dataset_path, and train_dataset_index_file in ```train.sh```. The train_dataset_index_file should be a .txt file, and each line should indicate an index of a training sample.
 
```txt
06269_062690093
27197_271975251
27197_271978467
...
```

Then you can use the following to run TextDiffuser:

```bash
accelerate launch train.py \
    --train_batch_size=24 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --mixed_precision="fp16" \
    --num_train_epochs=2 \
    --learning_rate=1e-5 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="experiment_name" \
    --enable_xformers_memory_efficient_attention \
    --dataloader_num_workers=4 \
    --character_aware_loss_lambda=0.01 \
    --resume_from_checkpoint="latest" \
    --drop_caption \
    --mask_all_ratio=0.5 \
    --segmentation_mask_aug \
    --dataset_path=/home/path/to/laion-ocr-unzip \
    --train_dataset_index_file=/path/to/index_file.txt \
    --vis_num=8
```

If you encounter an "out-of-memory" error, please consider reducing the batch size appropriately.
 
 
## :firecracker: Inference

TextDiffuser can be applied on: text-to-image, text-to-image-with-template, and text-inpainting.

### Text-to-Image
This task is designed to generate images based on given prompts. Users are required to enclose the keywords to be drawn with single quotation marks.

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode="text-to-image" \
  --resume_from_checkpoint="textdiffuser-ckpt/diffusion_backbone" \
  --prompt="A sign that says 'Hello'" \
  --output_dir="./output" \
  --vis_num=4
```

### Text-to-Image-with-Template
This task aims to generate images based on given prompts and template images (can be printed, handwritten, or scene text images). A pre-trained character-level segmentation model is used to extract layout information from the template image.

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode="text-to-image-with-template" \
  --resume_from_checkpoint="textdiffuser-ckpt/diffusion_backbone" \
  --prompt="a poster of monkey music festival" \
  --template_image="assets/examples/text-to-image-with-template/case2.jpg" \
  --output_dir="./output" \
  --vis_num=4
```

### Text-Inpainting
This task aims to modify a given image in an inpainting manner. The provided text mask image should contain the inpainting region and the text to be drawn within the region.

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode="text-inpainting" \
  --resume_from_checkpoint="textdiffuser-ckpt/diffusion_backbone" \
  --prompt="a boy draws good morning on a board" \
  --original_image="assets/examples/text-inpainting/case2.jpg" \
  --text_mask="assets/examples/text-inpainting/case2_mask.jpg" \
  --output_dir="./output" \
  --vis_num=4
```

## :chart_with_upwards_trend:	Evaluation

For evaluation, please download [MARIOEval](https://layoutlm.blob.core.windows.net/textdiffuser/MARIOEval.zip?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) and the generation results of each methods are at [link](https://layoutlm.blob.core.windows.net/textdiffuser/marioeval_generation.zip?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) for reference.
. MARIOEval contains 5,414 prompts for evaluation, including the following subsets:

| Subset | #Sample | Subset | #Sample |
| --- | ---: | --- | ---: |
| LAIONEval4000 | 4,000 | ChineseDrawText | 175 |
| TMDBEval500 | 500 | DrawBenchText | 21 |
| OpenLibrary500 | 500 | DrawTextCreative | 218 |

The structure of each folder is as follows:

```bash
├── LAIONEval4000/
│   ├── images/                       # ground truth images
│   ├── render/                       # layouts of keywords generated by Layout Transformer
│   ├── LAIONEval4000.txt             # prompts with keywords enclosed with quotes
│   └── LAIONEval4000_wo_quote.txt    # prompts without quotes
```

Please note that the ground truth images are only available for the LAIONEval4000, TMDBEval500, and OpenLibrary500 subsets. The render images are used for evaluating ControlNet. We manually enclose keywords with quotes according to the ocr results. Please refer to the ```_wo_quote.txt``` version for original prompts.

To evaluate TextDiffuser, please use the following command for sampling:

```python
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --mode="text-to-image" \
  --resume_from_checkpoint="textdiffuser-ckpt/diffusion_backbone" \
  --prompt_list="/path/to/MARIOEval/TMDBEval500/TMDBEval500.txt" \
  --output_dir="/path/to/output_dir" \
  --vis_num=4
```

To sample from other baseline methods (e.g, Stable Diffusion, ControlNet, and DeepFloyd), the scripts are provided in the ```./eval``` folder. We also provided the scripts for calculating FID, Clip Score, as well as the OCR metrics.


| Metrics | Stable Diffusion | ContolNet | DeepFloyd | TextDiffuser (Ours) |
| :---: | :---: | :---: | :---: | :---: |
| FID↓ | 51.295 | 51.485 | **34.902** | 38.758 |
| CLIPScore↑ | 0.3015 | 0.3424 | 0.3267 | **0.3436** |
| OCR-Accuracy↑ | 0.0003 | 0.2390 | 0.0262 | **0.5609** |
| OCR-Precision↑ | 0.0173 | 0.5211 | 0.1450 | **0.7846** |
| OCR-Recall↑ | 0.0280 | 0.6707 | 0.2245 | **0.7802** |
| OCR-Fmeasure↑ | 0.0214 | 0.5865 | 0.1762 | **0.7824** |
| *OCR-Accuracy↑ | 0.0178 | 0.2705 | 0.0457 | **0.5712** |
| *OCR-Precision↑ | 0.0192 | 0.5391 | 0.1738 | **0.7795** |
| *OCR-Recall↑ | 0.0260 | 0.6438 | 0.2235 | **0.7498** |
| *OCR-Fmeasure↑ | 0.0221 | 0.5868 | 0.1955 | **0.7643** |

Please note that OCR metrics begin with "\*" mean we use open-source [MaskTextSpotterV3](https://github.com/MhLiao/MaskTextSpotterV3) for evaluation, and without "\*" denote we use [MicroSoft OCR API](https://azure.microsoft.com/en-us/updates/computer-vision-v3-preview-6/) for evaluation. The performance of text-to-image on MARIO-Eval compared with existing methods. TextDiffuser performs
the best regarding CLIPScore and OCR evaluation while achieving comparable performance on FID.

<img src="assets/readme_images/userstudy.jpg" width="90%">

User studies for whole-image generation and part-image generation tasks. (a) For whole-image generation, our method clearly outperforms others in both aspects of text rendering quality and image-text matching. (b) For part-image generation, our method receives high scores from human evaluators in these two aspects.


## :joystick:	Demo
TextDiffuser has been deployed on [Hugging Face](https://huggingface.co/spaces/JingyeChen22/TextDiffuser). If you have advanced GPUs, you may deploy the demo locally as follows:

```python
CUDA_VISIBLE_DEVICES=0 python gradio_app.py
```

Then you can enjoy the demo with local browser:  

<img src="assets/readme_images/demo.jpg" width="90%">




## :framed_picture:	Gallery

### Text-to-Image
<img src="assets/readme_images/gallery_text-to-image.jpg" width="80%">

### Text-to-Image-with-Template
<img src="assets/readme_images/gallery_text-to-image-with-template.jpg" width="80%">

### Text-Inpainting
<img src="assets/readme_images/gallery_text-inpainting.jpg" width="80%">

## :love_letter: Acknowledgement

We sincerely thank the following projects: [Hugging Face Diffuser](https://github.com/huggingface/diffusers), [LAION](https://laion.ai/laion-400-open-dataset/), [DB](https://github.com/MhLiao/DB), [PARSeq](https://github.com/baudm/parseq), [img2dataset](https://github.com/rom1504/img2dataset).

Also, special thanks to the open-source diffusion project or available demo: [DALLE](https://openai.com/product/dall-e-2), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Stable Diffusion XL](https://dreamstudio.ai/generate), [Midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F), [ControlNet](https://github.com/lllyasviel/ControlNet), [DeepFloyd](https://github.com/deep-floyd/IF).

  
## :exclamation: Disclaimer
Please note that the code is intended for academic and research purposes **ONLY**. Any use of the code for generating inappropriate content is **strictly prohibited**. The responsibility for any misuse or inappropriate use of the code lies solely with the users who generated such content, and this code shall not be held liable for any such use.

## :envelope: Contact

For help or issues using TextDiffuser, please email Jingye Chen (qwerty.chen@connect.ust.hk), Yupan Huang (huangyp28@mail2.sysu.edu.cn) or submit a GitHub issue.

For other communications related to TextDiffuser, please contact Lei Cui (lecu@microsoft.com) or Furu Wei (fuwei@microsoft.com).

## :herb: Citation
If you find this code useful in your research, please consider citing:
```
@article{chen2023textdiffuser,
  title={TextDiffuser: Diffusion Models as Text Painters},
  author={Chen, Jingye and Huang, Yupan and Lv, Tengchao and Cui, Lei and Chen, Qifeng and Wei, Furu},
  journal={arXiv preprint arXiv:2305.10855},
  year={2023}
}
```
