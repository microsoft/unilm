## Setup

1. Clone the repo:
```bash
git clone https://github.com/microsoft/unilm.git
cd unilm/kosmos-2
```
2. Create conda env and install packages:
```bash
conda create -n kosmos-2 python=3.9
conda activate kosmos-2
pip3 install -r requirements.txt
```
3. Install [NVIDIA/apex](https://github.com/NVIDIA/apex/tree/master#linux)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

4. Install the packages:
```bash
bash vl_setup_xl.sh
``` 