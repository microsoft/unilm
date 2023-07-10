import json
from tqdm import tqdm
  
def cook_data(input_file, image_path, locate_token=None, postfix='.out'):  
    # read a json file
    print(input_file)
    dataset = json.load(open(input_file, 'r', encoding='utf-8'))
    with open(input_file + postfix, 'w', encoding='utf-8') as f:  
        for ann in tqdm(dataset['annotations']):  
            image_id = ann['image_id']  
            image_info = [img for img in dataset['images'] if img['id'] == image_id][0]  
            file_name = image_info['file_name']  
            caption = image_info['caption']  
            # pdb.set_trace()
            if 'train2014' in file_name:
                dir_name = 'train2014'
            else:
                dir_name = 'val2014'
            if not locate_token:  
                sentence = f'[image]{image_path}/{dir_name}/{file_name}<tab><phrase>{caption}</phrase>'  
            else:  
                sentence = f'[image]{image_path}/{dir_name}/{file_name}<tab>{locate_token}<phrase>{caption}</phrase>'  
  
            f.write(sentence + '\n') 
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help='/path/to/mdetr_annotations')
    parser.add_argument('image_path', help='/path/to/MSCOCO2014 images')
    args = parser.parse_args()
    
    cook_data(f"{args.json_path}/finetune_refcoco_testA.json", args.image_path, locate_token='<grounding>', postfix='.locout')
    cook_data(f"{args.json_path}/finetune_refcoco_testB.json", args.image_path, locate_token='<grounding>', postfix='.locout')
    cook_data(f"{args.json_path}/finetune_refcoco_val.json", args.image_path, locate_token='<grounding>', postfix='.locout')
    cook_data(f"{args.json_path}/finetune_refcoco+_testA.json", args.image_path, locate_token='<grounding>', postfix='.locout')
    cook_data(f"{args.json_path}/finetune_refcoco+_testB.json", args.image_path, locate_token='<grounding>', postfix='.locout')
    cook_data(f"{args.json_path}/finetune_refcoco+_val.json", args.image_path, locate_token='<grounding>', postfix='.locout')
    cook_data(f"{args.json_path}/finetune_refcocog_test.json", args.image_path, locate_token='<grounding>', postfix='.locout')
    cook_data(f"{args.json_path}/finetune_refcocog_val.json", args.image_path, locate_token='<grounding>', postfix='.locout')
    