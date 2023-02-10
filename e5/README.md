# E5 Text Embeddings

[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533.pdf).
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, Furu Wei, arXiv 2022

## Pre-trained Models

|          | # of layers | embedding dimension | Huggingface                                                                |                              OneDrive link                               |
|----------|:-----------:|:-------------------:|----------------------------------------------------------------------------|:------------------------------------------------------------------------:|
| E5-small |     12      |         384         | [intfloat/e5-small](https://huggingface.co/intfloat/e5-small)              |  [e5-small](https://1drv.ms/u/s!Ap3CZfrY6o7cgSRb_1wwJ2sxw34f?e=32vp1D)   |
| E5-base  |     12      |         768         | [intfloat/e5-base](https://huggingface.co/intfloat/e5-base)                |   [e5-base](https://1drv.ms/u/s!Ap3CZfrY6o7cgSbGDfaT56j1Wf0Q?e=huiOWd)   |
| E5-large |     24      |        1024         | [intfloat/e5-large](https://huggingface.co/intfloat/e5-large)              |  [e5-large](https://1drv.ms/u/s!Ap3CZfrY6o7cgSUQ-UWcZxBO1Sig?e=Tg30wc)   |
| E5-small-unsupervised |     12      |         384         | [intfloat/e5-small-unsupervised](https://huggingface.co/intfloat/e5-small-unsupervised) |     |
| E5-base-unsupervised  |     12      |         768         | [intfloat/e5-base-unsupervised](https://huggingface.co/intfloat/e5-base-unsupervised)                |      |
| E5-large-unsupervised |     24      |        1024         | [intfloat/e5-large-unsupervised](https://huggingface.co/intfloat/e5-large-unsupervised)              |     |

The models with `-unsupervised` suffix only pre-trains on unlabeled datasets.

## Install Python Package Requirements

```shell
pip install -r requirements.txt
```

## Evaluate on the [BEIR Benchmark](https://arxiv.org/abs/2104.08663)

After installing the required python packages,
run the following command on GPU machines:

```shell
bash scripts/eval_mteb_beir.sh intfloat/e5-small
```

You can also change the model name to local directory that stores the downloaded E5 model from OneDrive.
By default,
the evaluation script will use all the available GPUs.

Caution: it could take quite a long time (~10 hours) due to corpus encoding.

## Evaluate on the [MTEB Benchmark](https://arxiv.org/abs/2210.07316)

Run the following command:

```shell
bash scripts/eval_mteb.sh intfloat/e5-small
```

## Citation

If you find our paper or models helpful, please consider cite as follows:

```
@article{wang2022text,
  title={Text Embeddings by Weakly-Supervised Contrastive Pre-training},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2212.03533},
  year={2022}
}
```

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
