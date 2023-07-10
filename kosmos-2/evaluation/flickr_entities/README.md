# Flickr30k Entities

## Results
- Results of Kosmos-2 on Flickr30K Entities val split.
   
|   Recall@k  |        all         |      animals       |      bodyparts      |      clothing      |    instruments     |       other        |       people       |       scene        |      vehicles      |
|-------------|--------------------|--------------------|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
|   Recall@1  | 0.778355158317744  | 0.9082217973231358 | 0.41404805914972276 | 0.708779443254818  | 0.7677419354838709 | 0.6470767356881851 | 0.8895578874935489 | 0.8062622309197651 | 0.8668639053254438 |
|   Recall@5  | 0.7924201482713227 | 0.9158699808795411 | 0.42513863216266173 | 0.7220556745182013 | 0.7870967741935484 | 0.6632155907429963 | 0.9072767933941166 | 0.8075668623613829 | 0.8698224852071006 |
|  Recall@10  | 0.7925587196009146 | 0.9158699808795411 | 0.42513863216266173 | 0.7220556745182013 | 0.7870967741935484 | 0.6635200974421437 | 0.9074488216067436 | 0.8075668623613829 | 0.8698224852071006 |

- Results of Kosmos-2 on Flickr30K Entities test split.

|   Recall@k  |        all         |      animals       |     bodyparts      |      clothing      |    instruments     |       other        |       people       |       scene        | vehicles |
|-------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|----------|
|   Recall@1  | 0.7871693943788413 | 0.916988416988417  | 0.4091778202676864 | 0.7354726799653079 | 0.7654320987654321 | 0.6505631298162419 | 0.8951555869872702 | 0.8221124150710315 |  0.9125  |
|   Recall@5  | 0.8011187072715973 | 0.9247104247104247 | 0.4130019120458891 | 0.7437120555073721 | 0.7777777777777778 | 0.6615293420272673 | 0.9192008486562943 | 0.8233477455219271 |  0.9125  |
|  Recall@10  | 0.8013949312892756 | 0.9247104247104247 | 0.4130019120458891 | 0.7441457068516912 | 0.7777777777777778 | 0.6615293420272673 | 0.9197312588401697 | 0.8233477455219271 |  0.9125  |

## Data preparation

### 1. Download image and annotations
* Download the original Flickr30k image dataset from the [Flickr30K webpage](http://shannon.cs.illinois.edu/DenotationGraph/).
* Download the original Flickr30k entities annotations from [Flickr30k annotations](https://github.com/BryanPlummer/flickr30k_entities).
* Download the [MDETR](https://github.com/ashkamath/mdetr) pre-processed annotations: [Pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1).

### 2. Convert data format

You can run the following command to convert the data format from [MDETR pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1):
```python
python cook_data.py /path/to/mdetr_annotations /path/to/flickr-images
```

Alternatively, you also can download the pre-processed files: [val split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/final_flickr_separateGT_val.json.inline.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D
) and [test split](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/final_flickr_separateGT_test.json.inline.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D
). Remember to replace the image path in our provided files with the specific image path on your machine.

## Evaluation

You can run the following command to evaluate Kosmos-2 on Flickr30k Entities:

```bash
cd unilm/kosmos-2

# val split
bash evaluation/grd-zeroshot-flickr.sh 0 32 /path/to/kosmos-2.pt /path/to/final_flickr_separateGT_val.json.inline.locout /path/to/final_flickr_separateGT_val.json /path/to/flickr30k_entities

# test split
bash evaluation/grd-zeroshot-flickr.sh 0 32 /path/to/kosmos-2.pt /path/to/final_flickr_separateGT_test.json.inline.locout /path/to/final_flickr_separateGT_test.json /path/to/flickr30k_entities
```
where `final_flickr_separateGT_val.json` can be found after downloading and uncompressing the [MDETR annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1), `final_flickr_separateGT_test.json.inline.locout` can be downloaded or generated in [here](#2-convert-data-format), and `/path/to/flickr30k_entities` is the path where you cloned the official [Flickr30k annotations](https://github.com/BryanPlummer/flickr30k_entities).


Alternatively, download our provided evaluation results ([val_split_ouput](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.flickr.val.inline.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) and [test_split_ouput](https://conversationhub.blob.core.windows.net/beit-share-public/kosmos-2/eval/kosmos2_inst.pt.flickr.test.inline.locout?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)) and then evaluate:
```python
python evaluation/flickr/flickr_entities_evaluate.py /path/to/eval_result --annotation_file /path/to/final_flickr_separateGT_val.json --flickr_entities_path /path/to/flickr30k_entities
```
`final_flickr_separateGT_val.json` can be found after downloading and uncompressing the [MDETR annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1); `/path/to/flickr30k_entities` is the path where you cloned official [Flickr30k annotations](https://github.com/BryanPlummer/flickr30k_entities).