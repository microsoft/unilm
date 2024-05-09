import tiktoken
from typing import List


class TiktokenTokenizer:
    def __init__(self,
        tiktoken_model: str,
        tokenizer_pad_to_multiple: int = 8,
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
    ):
        self.symbols = [bos, pad, eos, unk]
        self.indices = {s: i for i, s in enumerate(self.symbols)}
        self.tokenizer_pad_to_multiple = tokenizer_pad_to_multiple
        cl100k_base = tiktoken.get_encoding(tiktoken_model)
        self._model = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name="cl100k_im",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={
                **cl100k_base._special_tokens,
                "<fim_prefix>": 100264,
                "<fim_middle>": 100265,
                "<fim_suffix>": 100266,
                "<fim_pad>": 100267,
                "<reponame>": 100268,
                "<filename>": 100269,
                "<gh_stars>": 100270,
                "<issue_start>": 100271,
                "<issue_comment>": 100272,
                "<issue_closed>": 100273,
                "<jupyter_start>": 100274,
                "<jupyter_text>": 100275,
                "<jupyter_code>": 100276,
                "<jupyter_output>": 100277,
                "<empty_output>": 100278,
                "<commit_before>": 100279,
                "<commit_msg>": 100280,
                "<commit_after>": 100281,
            }
        )

    @property
    def n_words(self) -> int:
        n_words = self._model.n_vocab + len(self.symbols)
        n_words = (n_words + self.tokenizer_pad_to_multiple - 1) // self.tokenizer_pad_to_multiple * self.tokenizer_pad_to_multiple
        return n_words

    @property
    def bos_id(self) -> int:
        return self.indices["<s>"]

    @property
    def eos_id(self) -> int:
        return self.indices["</s>"]

    @property
    def pad_id(self) -> int:
        return self.indices["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.indices["<unk>"]

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert isinstance(s, str)
        t = self._model.encode(s, allowed_special="all")
        t = [i + len(self.symbols) for i in t]
        if bos:
            t = [self.bos_id, *t]
        if eos:
            t = [*t, self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        t = [i - len(self.symbols) for i in t if i >= len(self.symbols)]
        return self._model.decode(t)