
import json
from tqdm import tqdm


def cook_data_inline(input_file, image_path, locate_token=None, postfix='.inline.out'):
    # read a json file
    obj = json.load(open(input_file, 'r', encoding='utf-8'))
    with open(input_file + postfix, 'w', encoding='utf-8') as f:
        for item in tqdm(obj['images']):
            file_name = item["file_name"]
            caption = item["caption"]
            if locate_token is None:
                sentence_prefix = f'[image]{image_path}/{file_name}<tab>'
            else:
                sentence_prefix = f'[image]{image_path}/{file_name}<tab>{locate_token}'
            postive_item_pos = item['tokens_positive_eval']

            for pos in postive_item_pos:
                if len(pos) > 1:
                    print("> 1", postive_item_pos, pos)
                pos_start, pos_end = pos[0]
                phrase = caption[pos_start:pos_end]
                prefix_caption = caption[:pos_start]
                f.write(sentence_prefix + f'{prefix_caption} <phrase>{phrase}</phrase>\n') 


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help='/path/to/mdetr_annotations')
    parser.add_argument('image_path', help='/path/to/flickr30k-images')
    args = parser.parse_args()
    
    cook_data_inline(f"{args.json_path}/final_flickr_separateGT_test.json", args.image_path, locate_token='<grounding>', postfix='.inline.locout')
    cook_data_inline(f"{args.json_path}/final_flickr_separateGT_val.json", args.image_path, locate_token='<grounding>', postfix='.inline.locout')
    