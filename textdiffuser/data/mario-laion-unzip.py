import json
import os
import traceback
from tqdm import tqdm
from multiprocessing import Pool

ROOT_FROM = 'XXX' # the path of laion-ocr-zip
ROOT_TO = 'XXX' # the path for saving dataset
MULTIPROCESSING_NUM = 64
DOWNLOAD_IMAGES = False  # whether to download images from urls

def unzip_file(idx):
    if not os.path.exists(f'{ROOT_FROM}/{idx}.zip') or os.path.exists(f'{ROOT_TO}/{idx}'):
        return
    cmd = f'unzip -q {ROOT_FROM}/{idx}.zip -d {ROOT_TO}'
    os.system(cmd)


def multiprocess_unzip_file(idxs):
    os.makedirs(ROOT_TO, exist_ok=True)

    with Pool(processes=MULTIPROCESSING_NUM) as p:
        with tqdm(total=len(idxs), desc='total') as pbar:
            for i, _ in enumerate(p.imap_unordered(unzip_file, idxs)):
                pbar.update()
    print("multiprocess_unzip_file done!")

if __name__ == '__main__':
    files = os.listdir(ROOT_FROM)
    idxs = [str(idx[:-4]).zfill(5) for idx in files]
    multiprocess_unzip_file(idxs)
    print("Finished!")
