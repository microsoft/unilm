# SpeechLM

<!--**Pre-trained models for speech related tasks**-->

 [**SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data**](https://arxiv.org/abs/2209.15329)


- (Done) Oct 2022: release the code and models
- Oct 2022: release preprint in [arXiv](https://arxiv.org/abs/2209.15329)

## Pre-Trained and Fine-tuned Models

|  Model            |               Pre-training Dataset                                                                            | Fine-tuning Dataset                                               | Model |
| :------:          | :----------------------------------------------:                                                              | :-----------------:                                               | :-----: |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      |                      -                                            | [Google drive](https://drive.google.com/file/d/1iJvhSGghNrMT-wAY1nwVu2YaYuTy1pxx/view?usp=sharing)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [100 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1mH3N7iKMWYk3rSBJErQPYf3x5ugqDq5x/view?usp=sharing)  |
| SpeechLM-H Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      |                      -                                            | [Google drive](https://drive.google.com/file/d/1eblW8U8f9t-NTuCNRrNHwr-8BeLAUAmQ/view?usp=sharing)  |
| SpeechLM-H Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [100 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1vXyO5DolbiWiTYZ6pkkKQsu2wJetaPlv/view?usp=sharing)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-De CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://valle.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_ende.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-Ca CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://valle.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_enca.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-Ar CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://valle.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_enar.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-Tr CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://valle.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_entr.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) |                      -                                            | [Google drive](https://drive.google.com/file/d/1QjLIgTJKIylVIp5hUkfSjGPtz8Xo7Lky/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [960 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1YZQDVv096o8Opt0RBnkRiZXYPRDqKZnP/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-De CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/1qYygNWSc11TQbBI1OzC4ChlR-dNh8t9S/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-Ca CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/162U88mwso2aVfzzPkEM2nP_vwTpcb57T/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-Ar CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/1lbTSRXewEeb2t45URunD6EiJcbniyjWW/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-Tr CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/1Er4I_jHS175pQQph223yKtiiLQ378VvH/view?usp=sharing)  |


## Extract features using pre-trained models
For easier use of our pre-trained models, we merge all inference-related code to [`SpeechLM.py`](SpeechLM.py) and make cleaned checkpoints [`SpeechLM-P Base`](https://msranlcmtteamdrive.blob.core.windows.net/share/speechlm/speechlmp_base_checkpoint_clean.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D) [`SpeechLM-H Base`](https://msranlcmtteamdrive.blob.core.windows.net/share/speechlm/speechlmh_base_checkpoint_clean.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D) [`SpeechLM-P Large`](https://msranlcmtteamdrive.blob.core.windows.net/share/speechlm/speechlmp_large_checkpoint_clean.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D) by removing non-required modules. Now you can directly use the following script to extract your speech features:
```python
import torch
import torch.nn.functional as F
from SpeechLM import SpeechLMConfig, SpeechLM

checkpoint = torch.load('path/to/the/cleaned/checkpoint.pt')
cfg = SpeechLMConfig(checkpoint['cfg']['model'])
model = SpeechLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

wav_input_16khz = torch.randn(1,10000)
normalize = checkpoint['cfg']['task']['normalize']  # False for base model, True for large model
if normalize:
    wav_input_16khz = F.layer_norm(wav_input_16khz[0], wav_input_16khz[0].shape).unsqueeze(0)

# extract the representation of last layer
rep = model.extract_features(wav_input_16khz)[0]

# extract the representation of each layer
output_layer = model.cfg.encoder_layers + model.cfg.text_transformer.encoder.layers
rep, layer_results = model.extract_features(wav_input_16khz, output_layer=output_layer, ret_layer_results=True)[0]
layer_reps = [x.transpose(0, 1) for x in layer_results]
```


## Setup
To fine-tune or pre-train more models, please follow the instructions below.

```bash
git submodule update --init SpeechLM/fairseq
cd SpeechLM/
pip install --editable fairseq/
pip install sacrebleu==1.5.1
```

## ASR on LibriSpeech
### Data preparation
Please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare `train.tsv` and `train.ltr`. You should make sure the vocabulary [`dict.ltr.txt`](dataset/LibriSpeech/asr/dict.ltr.txt) is the same as that used for the pre-trained model.

Put yout prepared data into `$data_dir`, we provided eamples in [`dataset/LibriSpeech/asr`](dataset/LibriSpeech/asr/). 

### Fine-tune a CTC model
- Fine-tune the base model
    ```bash
    # Usage: speechlm/scripts/tune_speechlm_asr/finetune_base_ctc.sh <model_path> <data_dir> <cpt_tag> [mount=$PWD] [world_size=8] [update_freq=1]
    model_path=path/to/your/pre-trained/model
    data_dir=dataset/LibriSpeech/asr
    bash speechlm/scripts/tune_speechlm_asr/finetune_base_ctc.sh $model_path $data_dir 'tag400k'
    ```
- Fine-tune the large model
    ```bash
    # Usage: speechlm/scripts/tune_speechlm_asr/finetune_large_ctc.sh <model_path> <data_dir> <cpt_tag> [mount=$PWD] [world_size=8] [update_freq=4]
    model_path=path/to/your/pre-trained/model
    data_dir=dataset/LibriSpeech/asr
    bash speechlm/scripts/tune_speechlm_asr/finetune_large_ctc.sh $model_path $data_dir 'tag400k'
    ```
### Decode
- Directly decode a CTC model.
    ```bash
    # Usage: speechlm/scripts/tune_speechlm_asr/inference_ctc.sh <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]
    model_path=path/to/your/fine-tuned/model
    data_dir=dataset/LibriSpeech/asr
    bash speechlm/scripts/tune_speechlm_asr/inference_ctc.sh $model_path $data_dir
    # for large models
    # bash speechlm/scripts/tune_speechlm_asr/inference_ctc_large.sh $model_path $data_dir
    ```
- Decode with 4-gram language model using [flashlight](https://github.com/flashlight/flashlight/tree/main/bindings/python) and [kenlm](https://github.com/kpu/kenlm).
    > Please put [4-gram.arpa](https://www.openslr.org/resources/11/4-gram.arpa.gz) and the word-to-letter lexicon [librispeech_lexicon.lst](https://drive.google.com/file/d/1q7IbNGqtwXnctjvuvpviQ4ZmepFHQmTO/view?usp=sharing) into `$data_dir`.
    ```bash
    # Usage: speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]
    model_path=path/to/your/fine-tuned/model
    data_dir=dataset/LibriSpeech/asr
    bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model_path $data_dir
    ```
- Decode large models with fairseq-lm using [flashlight](https://github.com/flashlight/flashlight/tree/main/bindings/python).
    > Please put [lm_librispeech_word_transformer.pt](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.pt) and its vocabulary [`dict.txt`](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.dict) into `$data_dir/fairseq_word_lm`, and the word-to-letter lexicon [librispeech_lexicon.lst](https://drive.google.com/file/d/1q7IbNGqtwXnctjvuvpviQ4ZmepFHQmTO/view?usp=sharing) into `$data_dir`. Capitalize the `dict.txt` to amke it compatible with the word-to-letter lexicon.
    ```bash
    # Usage: speechlm/scripts/tune_speechlm_asr/inference_ctc_large_fsqlm.sh <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]
    model_path=path/to/your/fine-tuned/model
    data_dir=dataset/LibriSpeech/asr
    bash speechlm/scripts/tune_speechlm_asr/inference_ctc_large_fsqlm.sh $model_path $data_dir dev_other
    ```

## ST on CoVoST-2
### Data Preparation
1. Download [Common Voice audio clips](https://commonvoice.mozilla.org/en/datasets) (version 4) for English into `$cv_root/en`.
2. Get data manifest. The following script will convert mp3 files to waveform, create tsv file containing speech/translation paires, create data config files.
    ```bash
    lang=de # ca,ar,tr
    cv_root=dataset/CommonVoice/v4
    bash speechlm/data_process/prepare_covost2_enxx.sh $lang $cv_root
    ```
    We provided examples in [`dataset/CommonVoice/v4/en/en-de`](dataset/CommonVoice/v4/en/en-de).

### Fine-tune a encoder-decoder model
- Fine-tune the Base model (fine-tuned models will be stored in `$mount/exp/finetune_covost`).

    ```bash
    model_path=path/to/your/pre-trained/model
    lang=de # ca,ar,tr
    data_dir=dataset/CommonVoice/v4/en/en-${lang}
    # Usage (Base model): speechlm/scripts/tune_speechlm_st/ft_base_covost_enxx.sh <model_path> <data_dir> <lang> <cpt-tag> [mount=$PWD] [world_size=8] [update_freq=2]
    bash speechlm/scripts/tune_speechlm_st/ft_base_covost_enxx.sh $model_path $data_dir $lang 'tag400k'
    ```
- Fine-tune the Large model (fine-tuned models will be stored in `$mount/exp/finetune_covost`).
    ```bash
    # Usage (Large model): speechlm/scripts/tune_speechlm_st/ft_large_covost_enxx.sh <model_path> <data_dir> <lang> <cpt-tag> [mount=$PWD] [world_size=8] [update_freq=4]
    bash speechlm/scripts/tune_speechlm_st/ft_large_covost_enxx.sh $model_path $data_dir $lang 'tag400k'
    ```

### Decode
- Decode the base model
    ```bash
    # Usage: speechlm/scripts/tune_speechlm_st/inference_base.sh <model_path> <data_dir> <lang> [gen-set=dev] [beam_size=5]
    model_path=path/to/your/fine-tuned/model
    lang=de # ca,ar,tr
    data_dir=dataset/CommonVoice/v4/en/en-${lang}
    bash speechlm/scripts/tune_speechlm_st/inference_base.sh $model_path $data_dir $lang dev
    ```
- Decode the large model
    ```bash
    # Usage: speechlm/scripts/tune_speechlm_st/inference_large.sh <model_path> <data_dir> <lang> [gen-set=dev] [beam_size=5]
    bash speechlm/scripts/tune_speechlm_st/inference_large.sh $model_path $data_dir $lang dev
    ```


## Pre-train
Please follow the instructions of [Tokenizer](README.md#Tokenizers) to prepare the pre-training data. We provided examples in [`dataset`](dataset).
- SpeechLM-P Base model

    Models will be stored in `$mount/pretrain`.
    ```bash
    data_dir=dataset/LibriSpeech/phone_unit   # should contain train_960.{tsv,phn}
    text_data_dir=dataset/LibriLM/phone_unit/bin-idx     # should contain train_text.phn-ltr.{phn,ltr}.{bin,idx}
    # Usage: speechlm/scripts/pretrain_speechlm/base_speechlmp.sh <data_dir> <text_data_dir> [mount=$PWD] [world_size=32] [update_freq=1]
    bash speechlm/scripts/pretrain_speechlm/base_speechlmp.sh $data_dir $text_data_dir
    ```
- SpeechLM-H Base model
    ```bash
    data_dir=dataset/LibriSpeech/hidden_unit  # should contain train_960.{tsv,phn}
    text_data_dir=dataset/LibriLM/km-ltr/bin-idx     # should contain train_text.km-ltr.{km,ltr}.{bin,idx}
    # Usage: speechlm/scripts/pretrain_speechlm/base_speechlmh.sh <data_dir> <text_data_dir> [mount=$PWD] [world_size=32] [update_freq=1]
    bash speechlm/scripts/pretrain_speechlm/base_speechlmp.sh $data_dir $text_data_dir
    ```
- SpeechLM-P Large model
    ```bash
    data_dir=dataset/LibriSpeech/phone_unit   # should contain train_960.{tsv,phn}
    text_data_dir=dataset/LibriLM/phone_unit/bin-idx     # should contain train_text.phn-ltr.{phn,ltr}.{bin,idx}
    # Usage: speechlm/scripts/pretrain_speechlm/base_speechlmp.sh <data_dir> <text_data_dir> [mount=$PWD] [world_size=32] [update_freq=1]
    bash speechlm/scripts/pretrain_speechlm/large_speechlmp.sh $data_dir $text_data_dir
    ```


## Tokenizers
### Phoneme-unit Tokenizer for Speech
This tokenizer is used to produce the frame-laigned phonemes for unlabeled speech, which is actually a hybrid HMM ASR model.

In the Base setting, we use 100h LibriSpeech labeled data to train the HMM model under Kaldi recipe, then decode the unpaired speech and get the aligned phonemes from the lattice.
Here we provided the processed phonemes of 960h speech here: [`train_960.tsv`](https://drive.google.com/file/d/1rxlikMglL2kEsF4NfqekZRoA02klY7CE/view?usp=sharing), [`train_960.phn`](https://drive.google.com/file/d/1pOzdB1qtofdgMohPnSUuOJVTPVFy2oRh/view?usp=sharing), [`dev_clean.tsv`](https://drive.google.com/file/d/1NuVwe687jLBFkDLRy1EV2A2uXyV_kBo2/view?usp=sharing), [`dev_clean.phn`](https://drive.google.com/file/d/1cq_gbS-UgCALOoaE5QmhWrhkTdXuc_Uc/view?usp=sharing). Note that the label-rate is 100 (10ms).

### Phoneme-unit Tokenizer for Text
This tokenizer is used to phonemize the unpaired text data to (phonemes, letters) paired data, following a `words -> phonemes -> upsampled phones` pipeline.

The following script will download LibriSpeech LM corpus and produce the required data: `train_text.phn-ltr.phn.{idx,bin}` and `train_text.phn-ltr.ltr.{idx,bin}`. 
> Before running it, make sure you have our provided [`dcit.phn.txt`](dataset/LibriLM/phone_unit/bin-idx/dcit.phn.txt) and [`dcit.ltr.txt`](dataset/LibriLM/phone_unit/bin-idx/dcit.ltr.txt) in the output dir `dataset/LibriLM/phone_unit/bin-idx/`.
```bash
# data will be in dataset/LibriLM/phone_unit/
bash speechlm/data_process/prepare_phn2ltr_librilm.sh
```
### Hidden-unit Tokenizer for Speech
Please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare 1) wav recordings [`train.tsv`](dataset/LibriSpeech/hidden_unit/train_sample100.tsv) and 2) corresponding hidden-units [`train.km`](dataset/LibriSpeech/hidden_unit/train_sample100.km), and 3) unit vocabulary [`dict.km.txt`](dataset/LibriSpeech/hidden_unit/dict.km.txt).

### Hidden-unit Tokenizer for Text
This tokenizer is used to produce the speech-style hidden units from unpaired text.
We train a [FastSpeech](https://arxiv.org/abs/2006.04558)-like model (instead generating continuous spectrum in the original paper, here we generate discrete units) on a small amount of ASR data ([100 hrs LibriSpeech](http://www.openslr.org/12)) as the tokenizer.

1. Convert asr transcripts to phoneme sequence.
2. Extract hidden-units from speech, using the [Hidden-unit Tokenizer for Speech](#hidden-unit-tokenizer-for-speech).
3. Train the [model](speechlm/models/fasttext2unit.py) on the paired data:
    ```bash
    data_dir=dataset/LibriSpeech/fast_phone2unit
    bash speechlm/scripts/tokenizer_fastT2U/train_s_5e-4.sh $data_dir
    ```
4. [Generate](speechlm/scripts/tokenizer_fastT2U/generate.sh) hidden units for a large text corpus:
    ```bash
    gen_set=dataset/LibriSpeech/fast_phone2unit/genset_examples
    bash speechlm/scripts/tokenizer_fastT2U/generate.sh $model_path $gen_set
    ```
We provided train/generate data examples in [`dataset/LibriSpeech/fast_phone2unit`](dataset/LibriSpeech/fast_phone2unit), and the model checkpoint [here](https://drive.google.com/file/d/1e-aYf8hPXuly8DEvNg5SISOlcUxsgED0/view?usp=sharing).

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq).

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

## Reference

If you find our work is useful in your research, please cite the following paper:

```bibtex
@article{zhang2022speechlm,
  title   = {SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data},
  author  = {Zhang, Ziqiang and Chen, Sanyuan and Zhou, Long and Wu, Yu and Ren, Shuo and Liu, Shujie and Yao, Zhuoyuan and Gong, Xun and Dai, Lirong and Li, Jinyu and Wei, Furu},
  eprint={2209.15329},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2022}
}
```

### Contact Information

For help or issues using SpeechLM models, please submit a GitHub issue.

For other communications related to SpeechLM, please contact Long Zhou (`lozhou@microsoft.com`).

