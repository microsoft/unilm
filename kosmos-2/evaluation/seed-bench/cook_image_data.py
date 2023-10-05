import base64
import io
import random
import os

# import pandas as pd
from PIL import Image
from tqdm import tqdm
import json
import string  

import pdb

# Example:
# json_ann['questions'][0]
# {
#   'answer': 'A', 
#   'choice_a': 'One', 
#   'choice_b': 'Two', 
#   'choice_c': 'Three', 
#   'choice_d': 'Four', 
#   'data_id': '1454426_2591111986', 
#   'data_type': 'image', 
#   'question': 'How many towels are in the image?', 
#   'question_id': '101669', 
#   'question_type_id': 5
# }

def convert_json_to_txt(json_path, image_path, txt_path, answer_path):
    json_ann = json.load(open(json_path, 'rb'))
    
    # pdb.set_trace()

    with open(txt_path, 'w', encoding='utf-8') as f, open(answer_path, 'w', encoding='utf-8') as fa:
        for index, item in tqdm(enumerate(json_ann['questions'])):
            if item['data_type'] != 'image':
                continue
            question = item['question'].replace('\n', ' ')
            image_file_path = os.path.join(image_path, item['data_id'])  

            options = {
                'A': item['choice_a'],
                'B': item['choice_b'],
                'C': item['choice_c'],
                'D': item['choice_d'],
            }
            for key, choice in options.items():
                choice = choice.replace('\n', ' ').strip()
                sentence = f'[image]{image_file_path}<tab>Question: {question} Answer: {choice}'  
                f.write(sentence + '\n')
                fa.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(index, question, choice, item['answer'], item['question_id'], item['question_type_id']))  

if __name__ == '__main__':
    save_dir = '/path/to/data'
    convert_json_to_txt(
                       f'{save_dir}/SEED-Bench/SEED-Bench.json', 
                       f'{save_dir}/SEED-Bench/SEED-Bench-image', 
                       f'{save_dir}/SEED-Bench/seed_bench_pplformat.txt',
                       f'{save_dir}/SEED-Bench/seed_bench_pplformat.answer'
                       )
