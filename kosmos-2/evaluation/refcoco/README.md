# RefCOCO, RefCOCO+, RefCOCOg

## Results
Zero-shot Results of Kosmos-2 on RefCOCO, RefCOCO+, RefCOCOg.
During the instruction tuning process, we removed all images from the RefCOCO/+/g val/test sets in the LLaVA-Instruct dataset to avoid image leakage (approximately 20K).

| Model | RefCOCO val | RefCOCO testA| RefCOCO testB | RefCOCO+ val | RefCOCO+ testA| RefCOCO+ testB | RefCOCOg val | RefCOCOg test|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kosmos-2 | 52.32 | 57.42 | 47.26 | 45.48 | 50.73 | 42.24 | 60.57 | 61.65 |

## Data preparation

### 1. Download image and annotations
* Download the official MSCOCO2014 image dataset from [here](http://images.cocodataset.org/zips/train2014.zip).
* Download the [MDETR](https://github.com/ashkamath/mdetr) pre-processed annotations: [Pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1).

### 2. Convert data format

You can run the following command to convert the data format from [MDETR pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1):
```python
python cook_data.py /path/to/mdetr_annotations /path/to/MSCOCO2014
```

Alternatively, you also can download the pre-processed files: 
[RefCOCO val split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/finetune_refcoco_val.json.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO testA split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/finetune_refcoco_testA.json.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO testB split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/finetune_refcoco_testB.json.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO+ val split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/finetune_refcoco+_val.json.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO+ testA split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/finetune_refcoco+_testA.json.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO+ testB split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/finetune_refcoco+_testB.json.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCOg val split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/finetune_refcocog_val.json.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) 
and [RefCOCOg test split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/finetune_refcocog_test.json.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
Remember to replace the image path in our provided files with the specific image path on your machine.

## Evaluation

You can run the following command to evaluate Kosmos-2 on these datasets:

```bash
cd unilm/kosmos-2

# RefCOCO val/testA/testB split
bash evaluation/grd-zeroshot-refcoco.sh 0 32 /path/to/kosmos-2.pt /path/to/finetune_refcoco_val.json.locout /path/to/finetune_refcoco_val.json
bash evaluation/grd-zeroshot-refcoco.sh 0 32 /path/to/kosmos-2.pt /path/to/finetune_refcoco_testA.json.locout /path/to/finetune_refcoco_testA.json
bash evaluation/grd-zeroshot-refcoco.sh 0 32 /path/to/kosmos-2.pt /path/to/finetune_refcoco_testB.json.locout /path/to/finetune_refcoco_testB.json

# RefCOCO+ val/testA/testB split
bash evaluation/grd-zeroshot-refcoco.sh 0 32 /path/to/kosmos-2.pt /path/to/finetune_refcoco+_val.json.locout /path/to/finetune_refcoco+_val.json
bash evaluation/grd-zeroshot-refcoco.sh 0 32 /path/to/kosmos-2.pt /path/to/finetune_refcoco+_testA.json.locout /path/to/finetune_refcoco+_testA.json
bash evaluation/grd-zeroshot-refcoco.sh 0 32 /path/to/kosmos-2.pt /path/to/finetune_refcoco+_testB.json.locout /path/to/finetune_refcoco+_testB.json

# RefCOCOg val/test split
bash evaluation/grd-zeroshot-refcoco.sh 0 32 /path/to/kosmos-2.pt /path/to/finetune_refcocog_val.json.locout /path/to/finetune_refcocog_val.json
bash evaluation/grd-zeroshot-refcoco.sh 0 32 /path/to/kosmos-2.pt /path/to/finetune_refcocog_test.json.locout /path/to/finetune_refcocog_test.json
```
where `finetune_refcoco*.json` can be found after downloading and uncompressing the [MDETR annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1), `finetune_*.json.locout` can be downloaded or generated in [here](#2-convert-data-format).

Alternatively, download our provided evaluation results [RefCOCO val result](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.refcoco.val.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO testA result](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.refcoco.testA.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO testB result](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.refcoco.testB.locout.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO+ val result](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.refcoco+.val.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO+ testA result](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.refcoco+.testA.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCO+ testB result](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.refcoco+.testB.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D), 
[RefCOCOg val result](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.refcocog.val.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) 
and [RefCOCOg test result](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.refcocog.test.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) and then evaluate:
```python
python evaluation/refcoco/refexp_evaluate.py /path/to/downloaded_results /path/to/finetune_refcoco*.json

# example
# python evaluation/refcoco/refexp_evaluate.py /tmp/kosmos2_inst.pt.refcoco.val.locout /tmp/mdetr_annotations/finetune_refcoco_val.json
```
where `finetune_refcoco*.json` can be found after downloading and uncompressing the [MDETR annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1); 
