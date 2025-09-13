CURRENT_DIR=$(pwd)
cd /tmp/
git clone https://github.com/QwenLM/Qwen2.5-Math.git
cd Qwen2.5-Math/evaluation/latex2sympy
pip install -e .
cd $CURRENT_DIR

pip install fairscale tensorboard optimum transformers tokenizers datasets einops opt_einsum iopath numpy==1.23.0 tiktoken boto3 sentencepiece scikit-learn huggingface-hub ninja wandb rouge-score xopen
pip install tqdm python_dateutil sympy==1.12 antlr4-python3-runtime==4.11.1 word2number Pebble timeout-decorator