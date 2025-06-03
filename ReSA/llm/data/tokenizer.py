import os
from typing import List
from transformers import LlamaTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Tokenizer:
    def __init__(self, tokenizer_path: str):
        self.tok = LlamaTokenizerFast.from_pretrained(tokenizer_path)

    @property
    def n_words(self) -> int:
        return self.tok.vocab_size

    @property
    def bos_id(self) -> int:
        return self.tok.encode(self.tok.bos_token)[-1]

    @property
    def eos_id(self) -> int:
        return self.tok.encode(self.tok.eos_token)[-1]

    @property
    def pad_id(self) -> int:
        return -100

    @property
    def unk_id(self) -> int:
        return self.tok.encode(self.tok.eos_token)[-1]

    def encode(self, s: str, bos: bool = True, eos: bool = False):
        tok = self.tok.encode(s, add_special_tokens=False)
        if bos:
            tok = [self.bos_id] + tok
        if eos:
            tok = tok + [self.eos_id]
        return tok
    
    def encode_batch(self, s: List[str], bos: bool = True, eos: bool = False):
        return [self.encode(s, bos, eos) for s in s]

    def decode(self, t: List[int]) -> str:
        t = [i for i in t if i != self.pad_id]
        return self.tok.decode(t, skip_special_tokens=True)

    def decode_batch(self, t: List[List[int]]) -> List[str]:
        t = [[i for i in x if i != self.pad_id] for x in t]
        return self.tok.batch_decode(t, skip_special_tokens=True)