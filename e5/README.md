# E5 Text Embeddings

[Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/pdf/2402.05672).
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei, arXiv 2024

[Improving Text Embeddings with Large Language Models](https://arxiv.org/pdf/2401.00368.pdf).
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei, arXiv 2024

[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533.pdf).
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, Furu Wei, arXiv 2022

## LLM based Models

|                        | BEIR | # of layers | embedding dimension | Huggingface                                                                             |
|------------------------|------|:-----------:|:-------------------:|-----------------------------------------------------------------------------------------|
| E5-mistral-7b-instruct | 56.9 |     32      |        4096         | [intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct)|

## English Pre-trained Models

|                        | BEIR | # of layers | embedding dimension | Huggingface                                                                             |
|------------------------|------|:-----------:|:-------------------:|-----------------------------------------------------------------------------------------|
| E5-small-v2            | 49.0 |     12      |         384         | [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2)                     |
| E5-base-v2             | 50.3 |     12      |         768         | [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)                       |
| E5-large-v2            | 50.6 |     24      |        1024         | [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2)                     |
|                        |      |             |                     |                                                                                         |
| E5-small               | 46.0 |     12      |         384         | [intfloat/e5-small](https://huggingface.co/intfloat/e5-small)                           |
| E5-base                | 48.8 |     12      |         768         | [intfloat/e5-base](https://huggingface.co/intfloat/e5-base)                             |
| E5-large               | 50.0 |     24      |        1024         | [intfloat/e5-large](https://huggingface.co/intfloat/e5-large)                           |
|                        |      |             |                     |                                                                                         |
| E5-small-unsupervised  | 40.8 |     12      |         384         | [intfloat/e5-small-unsupervised](https://huggingface.co/intfloat/e5-small-unsupervised) |
| E5-base-unsupervised   | 42.9 |     12      |         768         | [intfloat/e5-base-unsupervised](https://huggingface.co/intfloat/e5-base-unsupervised)   |
| E5-large-unsupervised  | 44.2 |     24      |        1024         | [intfloat/e5-large-unsupervised](https://huggingface.co/intfloat/e5-large-unsupervised) |

The models with `-unsupervised` suffix only pre-trains on unlabeled datasets.

## Multilingual Pre-trained Models

|                                | BEIR | # of layers | embedding dimension | Huggingface                                                                                               |
|--------------------------------|------|:-----------:|:-------------------:|-----------------------------------------------------------------------------------------------------------|
| multilingual-e5-small          | 46.6 |     12      |         384         | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)                   |
| multilingual-e5-base           | 48.9 |     12      |         768         | [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)                     |
| multilingual-e5-large          | 51.4 |     24      |        1024         | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)                   |
| multilingual-e5-large-instruct | 52.5 |     24      |        1024         | [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) |

## Install Python Package Requirements

```shell
pip install -r requirements.txt
```

For `e5-mistral-7b-instruct`, it would require `transformers>=4.34` to load Mistral model.

## Evaluate on the [BEIR Benchmark](https://arxiv.org/abs/2104.08663)

After installing the required python packages,
run the following command on GPU machines:

```shell
bash scripts/eval_mteb_beir.sh intfloat/e5-small-v2
```

By default,
the evaluation script will use all the available GPUs.

Caution: it could take quite a long time (~10 hours) due to corpus encoding.
For `intfloat/e5-mistral-7b-instruct`, it could take even longer (several days).

## Evaluate on the [MTEB Benchmark](https://arxiv.org/abs/2210.07316)

Run the following command:

```shell
bash scripts/eval_mteb_except_retrieval.sh intfloat/e5-small-v2
```

For multilingual models, simply add a `--multilingual` suffix:

```shell
bash scripts/eval_mteb_except_retrieval.sh intfloat/multilingual-e5-base --multilingual
```

## Other Resources

The data for our proposed synthetic task _personalized passkey retrieval_ is available at [https://huggingface.co/datasets/intfloat/personalized_passkey_retrieval](https://huggingface.co/datasets/intfloat/personalized_passkey_retrieval).

## Troubleshooting

If you encounter OOM error, please try to reduce the batch size.

## Citation

If you find our paper or models helpful, please consider cite as follows:

```
@article{wang2024multilingual,
  title={Multilingual E5 Text Embeddings: A Technical Report},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2402.05672},
  year={2024}
}

@article{wang2023improving,
  title={Improving Text Embeddings with Large Language Models},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2401.00368},
  year={2023}
}

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
