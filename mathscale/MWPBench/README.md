# MWPBench

#### [[Paper]](https://arxiv.org/abs/2403.02884)

We introduce MWPBench (stands for Math Word Problem Bench), a comprehensive and unified benchmark for math instruction tuned models. MWPBench amalgamates various datasets—GSM8K, MATH, TAL-SCQ, Math23k, Ape210k, GaokaoBench-Math, AGIEval series—covering a comprehensive range of mathematical education levels. We also present CollegeMath, a novel dataset derived from open-source college textbooks, to fill the gap in higher-education mathematical evaluation. Moreover, MWPBench standardizes evaluations across all datasets with a unified protocol, promoting equitable and reproducible model comparisons. Details can be found in [paper](https://arxiv.org/abs/2403.02884)

## Dataset Details

MWPBench comprises 20K training data points and 18K test data points. Each question in this dataset is paired with a concise answer for easy verification. Additionally, we have included the latest 30 math problems from the 2023 Gaokao Math exam. This new dataset is referred to as Fresh-GaokaoMath-2023. It is important to note that each data point is released in accordance with their original license and is intended solely for research purposes.

The raw data files for this project are available at the following locations:

- `data/full_train.json`
- `data/full_test.json`
- `data/fresh_gaokao_math_2023.json`

## Usage

Before initiating the evaluation process, users are advised to set up their environment following [open-instruct](https://github.com/allenai/open-instruct/blob/main/requirements.txt). Essential requirements for this setup include `torch`, `transformers`, `vllm`, and `openai`. To facilitate this setup, we have made available our environment configuration in `Dockerfile` and `requirements.txt` file.

To evaluate instruction tuned LLM on MWPBench test set, run:

```bash
cd MWPBench

python -m eval_vllm.driver \
    --data_file data/full_test.json \
    --model_name_or_path <local_path_to_your_model> \
    --batch_size 60 \
    --tensor_parallel_size <number_of_gpus> \
    --prompt_template "alpaca"
```

To evaluate ChatGPT on MWPBench test set, run:

```bash
cd MWPBench

python -m eval_openai.driver \
    --data_file data/full_test.json \
    --openai_model gpt-4-0314 \
    --num_threads 5 \
    --prompt_template "alpaca_force_ans" \
    --verbose
```

You can also experiment with different test sets. For instance, you can use the `data/fresh_gaokao_math_2023.json` file as an alternative by specifying it in the `data_file` argument, and manually check the extracted answers. Sample scripts are available in the `scripts` directory.

## Acknowledgments

We would like to thank the following projects for their contributions to our fuzzy matching code:

- [crfm-helm](https://github.com/stanford-crfm/helm)
- [WizardMath](https://github.com/nlpxucan/WizardLM/tree/main/WizardMath)

Their work has been invaluable to our project.

## Citation

```
@article{tang2024mathscale,
  title={MathScale: Scaling Instruction Tuning for Mathematical Reasoning},
  author={Tang, Zhengyang and Zhang, Xingxing and Wan, Benyou and Wei, Furu},
  journal={arXiv preprint arXiv:2403.02884},
  year={2024}
}
```
