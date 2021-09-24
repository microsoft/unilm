import os
from data import SROIETask2
from tqdm import tqdm
import shutil
import zipfile

if __name__ == '__main__':
    test_dir = '../SROIE_Task2_Original/test'
    output_dir = 'temp'
    os.makedirs(output_dir, exist_ok=True)
    generate_txt_path = '../generate-test.txt'
    output_file = None
    output_fp = None

    with open(generate_txt_path, 'r', encoding='utf8') as fp:
        lines = list(fp.readlines())
    while not lines[0].startswith('T-0'):
        lines = lines[1:]
    
    _, data = SROIETask2(test_dir, None, None)
    for t in tqdm(data):
        file_name = t['file_name']
        image_id = int(t['image_id'])

        this_output_file = os.path.basename(file_name).replace('.jpg', '.txt')
        if this_output_file != output_file:
            if output_fp is not None:
                output_fp.close()
            output_file = this_output_file
            output_fp = open(os.path.join(output_dir, output_file), 'w', encoding='utf8')

        pred_line_id = image_id * 4 + 2
        pred_line = lines[pred_line_id]
        assert pred_line.startswith('D-{:d}'.format(image_id))
        pred_line = pred_line[pred_line.find('\t') + 1:]
        pred_str = pred_line[pred_line.find('\t') + 1:]

        for word in pred_str.split():
            output_fp.write(word + '\n')
    
    if output_fp:
        output_fp.close()
        
    zip_fp = zipfile.ZipFile('predictions.zip', 'w')
    for txt_file in os.listdir(output_dir):
        zip_fp.write(os.path.join(output_dir, txt_file), txt_file)
    zip_fp.close()
    shutil.rmtree(output_dir)

