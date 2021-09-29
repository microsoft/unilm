import os
import logging
import sentencepiece as spm
from transformers.tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class XLMRTokenizer(PreTrainedTokenizer):

  def __init__(self, bpe_file, dict_file, **kwargs):
    super(XLMRTokenizer, self).__init__(
      bos_token="<s>",
      eos_token="</s>",
      unk_token="<unk>",
      pad_token="<pad>",
      mask_token="<mask>",
      sep_token="</s>",
      cls_token="<s>",
      **kwargs)
    
    self.max_len_single_sentence = self.max_len - 2
    self.max_len_sentences_pair = self.max_len - 4
    
    self.sp = spm.SentencePieceProcessor()
    self.sp.Load(bpe_file)

    self.encoder = {}
    self.decoder = []

    for token in [self.bos_token, self.pad_token, self.eos_token, self.unk_token]:
      self._add_token(token)
    
    with open(dict_file, encoding="utf-8") as fp:
      for line in fp:
        # NOTE DO NOT USE .split()
        tokens_cnt = line.rstrip().split(" ")
        try:
          assert len(tokens_cnt) >= 2, line
        except AssertionError:
          logger.error(
            "tokenizer line %s asserterror, replaced as <unk-%d>" % (
              line, len(self.decoder)))
          exit(0)
        self._add_token(" ".join(tokens_cnt[:-1]))
  
  def _add_token(self, token):
    idx = len(self.encoder)
    self.encoder[token] = idx
    self.decoder.append(token)

  def _tokenize(self, text):
    return self.sp.EncodeAsPieces(text)
  
  def _convert_id_to_token(self, index):
    return self.decoder[index]

  def _convert_token_to_id(self, token):
    return self.encoder.get(token, self.encoder.get(self.unk_token))

  def convert_tokens_to_string(self, tokens):
    return "".join(tokens).replace('\u2581', ' ').strip()
  
  @classmethod
  def from_pretrained(cls, model_path, **kwargs):
    bpe_file = os.path.join(model_path, "sentencepiece.bpe.model")
    dict_file = os.path.join(model_path, "dict.txt")
    tokenizer = cls(bpe_file, dict_file)
    return tokenizer
  
  def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if token_ids_1 is None:
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
    cls = [self.cls_token_id]
    sep = [self.sep_token_id]
    return cls + token_ids_0 + sep + sep + token_ids_1 + sep
  
  def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
    if already_has_special_tokens:
      if token_ids_1 is not None:
        raise ValueError("You should not supply a second sequence if the provided sequence of ids is already formated with special tokens for the model.")
      return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

    if token_ids_1 is None:
      return [1] + ([0] * len(token_ids_0)) + [1]
    return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

  def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
    sep = [self.sep_token_id]
    cls = [self.cls_token_id]

    if token_ids_1 is None:
      return len(cls + token_ids_0 + sep) * [0]
    return len(cls + token_ids_0 + sep) * [0] + len(sep + token_ids_1 + sep) * [1]


if __name__ == "__main__":  
  tokenizer = XLMRTokenizer.from_pretrained("/home/v-zechi/data/unilm/zechi/exp/bert_data/xlmr-large")
  
  for text in ["Hello world!", "你好，世界", "नमस्ते दुनिया", "مرحبا بالعالم", "Bonjour le monde"]:
    print(tokenizer.tokenize(text))
    print(tokenizer.encode_plus(text, text, add_special_tokens=True))
