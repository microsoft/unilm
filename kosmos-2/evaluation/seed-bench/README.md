# [SEED-Bench](https://github.com/AILab-CVC/SEED-Bench)

## Results
Zero-shot Results of Kosmos-2 on SEED-Bench are shown [here](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard)

## Data preparation

1. Download image and annotations follow [official instruction](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md).
2. Run `kosmos-2/evaluation/seed-bench/cook_image_data.py` to convert the data format (Image task only now).
3. Run the following command to eval:
```python
cd unilm/kosmos-2

bash evaluation/zeroshot-seed-bench.sh 0 32 /path/to/kosmos-2.pt /path/to/seed-bench/converted-data
```