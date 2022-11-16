import random
import numpy as np
import torch
import os
import shutil
# import logging
import sys


# def set_logging(args):
#     '''
#     Set logger for recording
#     '''
#     logging.basicConfig(filename="./output/{}/log.txt".format(args.exp_name), level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))



def set_seed(args):
    '''
    Set seed for reproducibility
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0: 
    #     torch.cuda.manual_seed_all(args.seed)


def set_exp_folder(args):
    '''
    Create a folder to store experimental results e.g., checkpoints or log
    '''
    os.makedirs(os.path.join(args.output_dir, 'output'), exist_ok=True) 

    if os.path.exists(os.path.join('output', args.exp_name)):
        if not args.overwrite_output_dir: 
            assert False, 'The exp_name is already used. Please modify the experiment name or use --overwrite_output_dir'
        else:
            print('Remove original directories.')
            shutil.rmtree(os.path.join('output', args.exp_name))
            print('Remove successfully.')
    
    os.makedirs(os.path.join(args.output_dir, 'output', args.exp_name), exist_ok=True)
    exp_path = os.path.join(args.output_dir, 'output', args.exp_name)
    print(f'Path [{exp_path}] has been created')


def check_screen():
    '''
    Check whether the experiment is in screen
    '''
    text = os.popen('echo $STY').readlines()
    string = ''
    for line in text:
        string += line
    if len(string.strip()) == 0:
        print("**** Attention Please! The code is not executed in Screen! ****")
    else:
        print(f'**** Screen Name : {string} ****')

    