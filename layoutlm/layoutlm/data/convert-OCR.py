import os, sys, subprocess, shutil

def convertImages(root):
    out_dir = root + "/out-OCR/"
    try:
        os.mkdir(out_dir) # make output directory if it does not exist already
    except:
        pass
    for path, directories, files in os.walk(root):
        for file in files:  # walk through dataset for img files
            if file.endswith('.png') | file.endswith('.jpg'):
                img = os.path.join(path, file)
                out = os.path.join(out_dir, os.path.splitext(file)[0])
                subprocess.run(
                    ["tesseract", img, out, "hocr"])  # make call to tesseract with img file and output directory
    for path, directories, files in os.walk(out_dir):
        for file in files:  # walk through output directory for .hocr files
            if file.endswith('.hocr'):
                base, ext = os.path.splitext(file)
                old_file_path = os.path.join(path, file)
                new_file_path = os.path.join(out_dir, base + '.xml')
                if not os.path.exists(new_file_path):
                    shutil.copy(old_file_path, new_file_path)  # copy .hocr file to .xml
                    os.remove(old_file_path)  # delete original .hocr file


if __name__ == "__main__":
    convertImages(sys.argv[1])