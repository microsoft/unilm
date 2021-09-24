import os 
import shutil
import zipfile

if __name__ == '__main__':
    input_dir = 'temp'
    output_dir = 'temp2'
    # os.makedirs(output_dir)

    for txt_file in os.listdir(input_dir):
        words = []
        single_colons = 0
        end_by_colons = 0
        with open(os.path.join(input_dir, txt_file), 'r') as fp:
            for word in fp.readlines():
                word = word.rstrip()
                if word == ':':
                    single_colons += 1                    
                elif word.endswith(':'):
                    end_by_colons += 1
                words.append(word)
        
        if end_by_colons > single_colons:
            fix_colon = True
        else:
            fix_colon = False
        
        if fix_colon:
            with open(os.path.join(output_dir, txt_file), 'w') as fp:
                for word in words:
                    if word != ':' and word.endswith(':'):
                        fp.write(word[:-1] + '\n:\n')
                    else:
                        fp.write(word + '\n')
        else:
            print(txt_file)
            shutil.copyfile(os.path.join(input_dir, txt_file), os.path.join(output_dir, txt_file))
        
    zip_fp = zipfile.ZipFile('predictions.zip', 'w')
    for txt_file in os.listdir(output_dir):
        zip_fp.write(os.path.join(output_dir, txt_file), txt_file)
    zip_fp.close()
    # shutil.rmtree(output_dir)