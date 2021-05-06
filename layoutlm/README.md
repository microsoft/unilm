# LayoutLM
**Multimodal (text + layout/format + image) pre-training for document AI**

- April 17th, 2021: [LayoutXLM](https://arxiv.org/abs/2104.08836) extends the LayoutLM/LayoutLMv2 into multilingual support! In addition, we also introduce XFUN, a multilingual form understanding benchmark including forms with human labeled key-value pairs in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese).
- December 29th, 2020: [LayoutLMv2](https://arxiv.org/abs/2012.14740) is coming with the new SOTA on a wide varierty of document AI tasks, including [DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1) and [SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) leaderboard.


## Introduction

LayoutLM is a simple but effective multi-modal pre-training method of text, layout and image for visually-rich document understanding and information extraction tasks, such as form understanding and receipt understanding. LayoutLM archives the SOTA results on multiple datasets. For more details, please refer to our paper: 

[LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)
Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, [KDD 2020](https://www.kdd.org/kdd2020/accepted-papers)

[LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740)
Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou, [ACL 2021](#)

[LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836)
Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei, [Preprint](#)

## Release Notes
**\*\*\*\*\* New Aug 7th, 2020: Our new document understanding datasets, [TableBank](https://doc-analysis.github.io/tablebank-page/) (LREC 2020) and [DocBank](https://doc-analysis.github.io/docbank-page/) (COLING 2020), are now publicly available.\*\*\*\*\***

**\*\*\*\*\* New May 16th, 2020: Our LayoutLM paper has been accepted to KDD 2020 as a full paper in the research track\*\*\*\*\***

**\*\*\*\*\* New Feb 18th, 2020: Initial release of pre-trained models and fine-tuning code for LayoutLM v1 \*\*\*\*\***

## Pre-trained Model

We pre-train LayoutLM on IIT-CDIP Test Collection 1.0\* dataset with two settings. 

* LayoutLM-Base, Uncased (11M documents, 2 epochs): 12-layer, 768-hidden, 12-heads, 113M parameters || [OneDrive](https://1drv.ms/u/s!ApPZx_TWwibInS3JD3sZlPpQVZ2b?e=bbTfmM) | [Google Drive](https://drive.google.com/open?id=1Htp3vq8y2VRoTAwpHbwKM0lzZ2ByB8xM)
* LayoutLM-Large, Uncased (11M documents, 2 epochs): 24-layer, 1024-hidden, 16-heads, 343M parameters || [OneDrive](https://1drv.ms/u/s!ApPZx_TWwibInSy2nj7YabBsTWNa?e=p4LQo1) | [Google Drive](https://drive.google.com/open?id=1tatUuWVuNUxsP02smZCbB5NspyGo7g2g)

\*As some downstream datasets are the subsets of IIT-CDIP, we have carefully excluded the overlap portion from the pre-training data.

## Fine-tuning Example

We evaluate LayoutLM on several document image understanding datasets, and it outperforms several SOTA pre-trained models and approaches.

Setup environment as follows:

~~~bash
conda create -n layoutlm python=3.6
conda activate layoutlm
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install .
## For development mode
# pip install -e ".[dev]"
~~~

### Sequence Labeling Task


We give a fine-tuning example for sequence labeling tasks. You can run this example on [FUNSD](https://guillaumejaume.github.io/FUNSD/), a dataset for document understanding tasks.

First, we need to preprocess the JSON file into txt. You can run the preprocessing scripts `funsd_preprocess.py` in the `scripts` directory. For more options, please refer to the arguments.

~~~bash
cd examples/seq_labeling
./preprocess.sh
~~~

After preprocessing, run LayoutLM as follows:

~~~bash
python run_seq_labeling.py  --data_dir data \
                            --model_type layoutlm \
                            --model_name_or_path path/to/pretrained/model/directory \
                            --do_lower_case \
                            --max_seq_length 512 \
                            --do_train \
                            --num_train_epochs 100.0 \
                            --logging_steps 10 \
                            --save_steps -1 \
                            --output_dir path/to/output/directory \
                            --labels data/labels.txt \
                            --per_gpu_train_batch_size 16 \
                            --per_gpu_eval_batch_size 16 \
                            --fp16
~~~

Note: The `DataParallel` will be enabled automatically to utilize all GPUs. If you want to train with `DistributedDataParallel`, please run the script like:

~~~bash
# Suppose you have 4 GPUs. 

python -m torch.distributed.launch --nproc_per_node=4 run_seq_labeling.py  --data_dir data \
                            --model_type layoutlm \
                            --model_name_or_path path/to/pretrained/model/directory \
                            --do_lower_case \
                            --max_seq_length 512 \
                            --do_train \
                            --num_train_epochs 100.0 \
                            --logging_steps 10 \
                            --save_steps -1 \
                            --output_dir path/to/output/directory \
                            --labels data/labels.txt \
                            --per_gpu_train_batch_size 16 \
                            --per_gpu_eval_batch_size 16 \
                            --fp16
~~~



Then you can do evaluation or inference by replacing `--do_train` with `--do_eval` or `--do_predict`

Also, you can run Bert and RoBERTa baseline by modifying the `--model_type` argument. For more options, please refer to the arguments of `run.py`.

### Document Image Classification Task

We also fine-tune LayoutLM on the document image classification task. You can download the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset from [here](https://www.cs.cmu.edu/~aharley/rvl-cdip/). Because this dataset only provides the document image, you should use the OCR tool to get the texts and bounding boxes. For example, you can easily use Tesseract, an open-source OCR engine, to generate corresponding OCR data in hOCR format. For more details, please refer to the [Tesseract wiki](https://github.com/tesseract-ocr/tesseract/wiki). Your processed data should look like [this sample data](https://1drv.ms/u/s!ApPZx_TWwibInTlBa5q3tQ7QUdH_?e=UZLVFw). 

With the processed OCR data, you can run LayoutLM as follows:

~~~bash
python run_classification.py  --data_dir  data \
                              --model_type layoutlm \
                              --model_name_or_path path/to/pretrained/model/directory \
                              --output_dir path/to/output/directory \
                              --do_lower_case \
                              --max_seq_length 512 \
                              --do_train \
                              --do_eval \
                              --num_train_epochs 40.0 \
                              --logging_steps 5000 \
                              --save_steps 5000 \
                              --per_gpu_train_batch_size 16 \
                              --per_gpu_eval_batch_size 16 \
                              --evaluate_during_training \
                              --fp16 
~~~

Similarly, you can do evaluation by changing `--do_train` to `--do_eval` and `--do_test`

Like the sequence labeling task, you can run Bert and RoBERTa baseline by modifying the `--model_type` argument.

### Results

#### SROIE (field-level)


| Model                                                        | Hmean      |
| ------------------------------------------------------------ | ---------- |
| BERT-Large                                                   | 90.99%     |
| RoBERTa-Large                                                | 92.80%     |
| [Ranking 1st in SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) | 94.02%     |
| [**LayoutLM**](https://rrc.cvc.uab.es/?ch=13&com=evaluation&view=method_info&task=3&m=71448) | **96.04%** |

#### RVL-CDIP

| Model                                                        | Accuracy   |
| ------------------------------------------------------------ | ---------- |
| BERT-Large                                                   | 89.92%     |
| RoBERTa-Large                                                | 90.11%     |
| [VGG-16 (Afzal et al., 2017)](https://arxiv.org/abs/1704.03557) | 90.97%     |
| [Stacked CNN Ensemble (Das et al., 2018)](https://arxiv.org/abs/1801.09321) | 92.21%     |
| [LadderNet (Sarkhel & Nandi, 2019)](https://www.ijcai.org/Proceedings/2019/0466.pdf) | 92.77%     |
| [Multimodal Ensemble (Dauphinee et al., 2019)](https://arxiv.org/abs/1912.04376) | 93.07%     |
| **LayoutLM**                                                 | **94.42%** |

#### FUNSD (field-level)

| Model         | Precision  | Recall     | F1         |
| ------------- | ---------- | ---------- | ---------- |
| BERT-Large    | 0.6113     | 0.7085     | 0.6563     |
| RoBERTa-Large | 0.6780     | 0.7391     | 0.7072     |
| **LayoutLM**  | **0.7677** | **0.8195** | **0.7927** |

## Citation

If you find LayoutLM useful in your research, please cite the following paper:

``` latex
@misc{xu2019layoutlm,
    title={LayoutLM: Pre-training of Text and Layout for Document Image Understanding},
    author={Yiheng Xu and Minghao Li and Lei Cui and Shaohan Huang and Furu Wei and Ming Zhou},
    year={2019},
    eprint={1912.13318},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using LayoutLM, please submit a GitHub issue.

For other communications related to LayoutLM, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).

