import os, sys, subprocess, shutil

def convert_all_to_xml(root):
    output_dir = root + '/out-OCR/'
    try:
        os.mkdir(output_dir) # make output directory if it does not exist already
    except:
        pass
    for path, directories, files in os.walk(root):
        for file in files:  # walk through dataset for img files
            if file.endswith('.png') | file.endswith('.jpg') | file.endswith('.tif'):
                img = os.path.join(path, file)
                convert_img_to_xml(img, output_dir)

def convert_img_to_xml(img, output_dir):
    img_path, img_file = os.path.split(img)
    output_file = os.path.join(output_dir, os.path.splitext(img_file)[0])
    subprocess.run(['tesseract', img, output_file, 'hocr']) # make call to Tesseract with img file and output dir
    old_file_path = output_file + '.hocr'
    new_file_path = output_file + '.xml'
    if not os.path.exists(new_file_path):
        shutil.copy(old_file_path, new_file_path) # copy .hocr file contents into .xml file
        os.remove(old_file_path) # delete original .hocr file


if __name__ == "__main__":
    convert_all_to_xml(sys.argv[1])