import json  
import hashlib  
import io  
import os  
import base64  
from PIL import Image  
from tqdm import tqdm

def calculate_md5(image):  
    md5_hash = hashlib.md5()  
    with io.BytesIO() as output:  
        image.save(output, format='JPEG')  
        image_data = output.getvalue()  
        md5_hash.update(image_data)  
    return md5_hash.hexdigest()  
  
def process_files(directory):  
    tsv_data = []  
  
    for file in tqdm(os.listdir(directory)):  
        if file.endswith('.json'):  
            json_path = os.path.join(directory, file)  
            jpg_path = os.path.join(directory, file.replace('.json', '.jpg'))  
  
            with open(json_path, 'r') as json_file:  
                data = json.load(json_file)  
  
            image = Image.open(jpg_path)  
            md5 = calculate_md5(image)  
            caption = data['caption']  
            width = data['width']  
            height = data['height']  
              
            with io.BytesIO() as buffer:  
                image.save(buffer, format='JPEG')  
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")  
  
            combined_data_str = {'phrase': data['noun_chunks'], 'expression_v1': data['ref_exps']}
  
            tsv_row = [md5, caption, image_base64, width, height, combined_data_str]  
            tsv_data.append('\t'.join(map(str, tsv_row)))  
  
    return tsv_data  
  
def write_tsv(tsv_data, output_file):  
    with open(output_file, 'w') as file:  
        file.write('\n'.join(tsv_data))  
  
if __name__ == '__main__':  
    directory = '/tmp/grit'  
    output_file = '/tmp/output.tsv'  
    tsv_data = process_files(directory)  
    write_tsv(tsv_data, output_file)  
