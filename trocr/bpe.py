from tempfile import tempdir
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig
from fairseq.data.encoders import register_bpe
import logging

logger = logging.getLogger(__name__)

INSERT_OR_REPLACE = 0 # 1 for replace and 0 for insert

@register_bpe("gpt2es", dataclass=GPT2BPEConfig) # as stands for attention space
class GPT2BPEEnhancedSpace(GPT2BPE):
    def __init__(self, cfg):
        logger.info('Using the GPT2BPEEnhancedSpace.')
        super().__init__(cfg)

    def encode(self, x: str) -> str:
        # only for sroie
        assert not x.startswith(' ')
        assert not x.endswith(' ')
        if INSERT_OR_REPLACE == 1:
            temp = []   
            word = ''                     
            for ch in x:
                if ch == ' ':
                    if word:
                        temp.append(word)
                        word = ''
                    temp.append('<s>')
                else:
                    word += ch
            if word:
                temp.append(word)

            for i in range(len(temp)):
                if temp[i] != '<s>':
                    temp[i] = ' '.join(map(str, self.bpe.encode(temp[i])))
                        
            return ' '.join(temp)
        elif INSERT_OR_REPLACE == 0:
            temp = []   
            word = ''                     
            for ch in x:
                if ch == ' ':
                    if word:
                        temp.append(word)
                        word = ' '
                    temp.append('<s>')
                else:
                    word += ch
            if word:
                temp.append(word)

            for i in range(len(temp)):
                if temp[i] != '<s>':
                    temp[i] = ' '.join(map(str, self.bpe.encode(temp[i])))
            
            return ' '.join(temp)           
                    
    def decode(self, x: str) -> str:
        if INSERT_OR_REPLACE == 1:            
            return self.bpe.decode(
                [int(tok) if tok not in {"<unk>", "<mask>", "<s>"} else tok for tok in x.split()]
            ).replace('<s>', ' ')
        elif INSERT_OR_REPLACE == 0:
            return self.bpe.decode(
                [int(tok) if tok not in {"<unk>", "<mask>", "<s>"} else tok for tok in x.split()]
            ).replace('<s>', '')

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")


if __name__ == '__main__':
    from data import SROIETask2
    config = GPT2BPEConfig()
    bpe = GPT2BPEEnhancedSpace(config)
    bpe2 = GPT2BPE(config)
    data, crop_data = SROIETask2('data/SROIE_Task2_Original/train', None, None, None)
    for item in crop_data:
        if item['image_id'] == 128:
            print()
        encoded_str = bpe.encode(item['encoded_str'])
        assert bpe.decode(encoded_str) == item['encoded_str'], '{:d} errors'.format(item['image_id'])
    # print(bpe.encode(' hello world!'))
    # print(bpe2.encode(' hello world!'))

