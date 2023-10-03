IMAGE_KEY="Images"
TEXT_KEY="Extracted"

import os, json, random, re

max_image_num = 5
tokens_per_sample = 2048
from spacy.lang.en import English

import sentencepiece as spm

nlp_sentencizer = English()
nlp_sentencizer.add_pipe("sentencizer")
spm_tokenizer = spm.SentencePieceProcessor(model_file=r"C:\Users\shaohanh\Desktop\sentencepiece.bpe.model")

def text_transform(line): 
    tokens = spm_tokenizer.encode(line, out_type=str)
    return tokens

def clean(text):
    # python re, remove html tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def _read_from_files(source_file):
    """
    <s> <image> image token </image> sentence <image> image token </image> sentence </s>
    1, sample a random subsequnece:  3 sentences + the first image ... take up to 5 images + 3 sentences
    2, filter html tags <p>, <br>, <br/>
    3, single image, random sample rate as 0.5
    """
    file_path = os.path.join(source_file)
    
    if not os.path.exists(file_path):
        print('| file {} not exists'.format(file_path), flush=True)
        return iter([]) # skip bad file

    try:
        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.read().strip().split('\n')
    except:
        return iter([]) # skip bad file

    for doc_jsonstr in lines:
        json_obj = json.loads(doc_jsonstr)
        doc = ['bos']
        start_idx = 0
        image_num = len(json_obj[IMAGE_KEY])
        if image_num == 1:
            r = random.random()
            if r > 0.5:
                continue
        for image_idx, image_item in enumerate(json_obj[IMAGE_KEY]):
            if image_idx >= max_image_num:
                yield doc
                break
            
            text_snippet = json_obj[TEXT_KEY][start_idx:image_item['Span'][0]-1]
            text_snippet = clean(text_snippet)
            if len(text_snippet) != 0:
                if image_idx == 0:
                    # crop 3 sentences before the first image
                    sentences = list(nlp_sentencizer(text_snippet).sents)
                    text_snippet = ' '.join([str(sent) for sent in sentences[-3:]])
                text_token = text_transform(text_snippet)
                doc.extend(text_token)
                if len(doc) >= tokens_per_sample: # drop too long sentence
                    # data.append(doc[:])
                    doc = doc[:tokens_per_sample - 2]
                    doc.append('eos')
                    yield doc
                    break
                    
            image_tokens = [i for i in image_item['input_ids']]
            doc.append('BOI_SYMBOL')
            doc.extend(image_tokens)
            doc.append('EOI_SYMBOL')
            
            start_idx = image_item['Span'][1] + 1
            if image_idx == image_num - 1:
                # crop 3 sentences after the last image
                text_snippet = json_obj[TEXT_KEY][start_idx:]
                text_snippet = clean(text_snippet)
                sentences = list(nlp_sentencizer(text_snippet).sents)
                text_snippet = ' '.join([str(sent) for sent in sentences[:3]])
                text_token = text_transform(text_snippet)
                doc.extend(text_token)
                doc.append('eos')
                if len(doc) < tokens_per_sample:
                    yield doc
                    break

all_length = []
image_num = []
token_length = []
j = 0
for item in _read_from_files(r"C:\Users\shaohanh\Desktop\partition.000.ndjson"):
    # all_length.append(len(item))
    # image_num.append(item.count('BOI_SYMBOL'))
    # token_length.append(len(item) - item.count('BOI_SYMBOL') * 197)
    print(item)
    j += 1
    if j > 10:
        break
    # if j % 1000 == 0:
    #     print(len(all_length), flush=True)
    #     print(j)


print('average length: ', sum(all_length) / len(all_length))
print('average image num: ', sum(image_num) / len(image_num))
print('average token length: ', sum(token_length) / len(token_length))