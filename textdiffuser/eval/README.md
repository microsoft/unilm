# Evaluation

We provide the code for sampling from Stable Diffusion, ControlNet, DeepFloyd at ```MARIOEval_generate.py```. Since these methods rely on *diffusers* of the original version, it is recommended to create a **NEW** environment and install packages with command ```pip install requirements.txt```. It is recommended to install pytorch with version >= 2.0 to avoid the OOM error.

Once the generation is complete, evaluation of FID and CLIPScore can be performed using the ```MARIOEval_evaluate.py``` file. For OCR metrics, please install [MaskTextSpotterV3](https://github.com/MhLiao/MaskTextSpotterV3) to obtain the OCR result of each image and refer to ```ocr_eval.py``` for evaluation. It should be noted that the output image of DeepFloyd contains a watermark "IF" at the right-bottom corner, which needs to be masked before performing OCR.

```python
if method is 'deepfloyd':
    image[-64:, -64:] = 0 # remove watermark, the input image is resized to 512x512
```
