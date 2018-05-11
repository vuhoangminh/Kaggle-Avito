import argparse
import pandas as pd
import sys, os
import textblob
from tqdm import tqdm,tqdm_pandas
import math
import mlcrate as mlc
import gc
import numpy as np
from googletrans import Translator
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import time
import nltk, textwrap

from lib.print_info import print_debug, print_doing, print_memory
from lib.read_write_file import save_csv, save_feather, save_file, save_pickle
from lib.read_write_file import load_csv, load_feather, load_pickle, read_train_test
from lib.gen_feature import generate_groupby_by_type_and_columns, create_time

SEED = 1988
cwd = os.getcwd()
if 'codes' not in cwd:
    default_path = cwd + '/codes/'
    os.chdir(default_path)

DATATYPE_DICT = {
    'count'     : 'uint32',
    'nunique'   : 'uint32',
    'cumcount'  : 'uint32',
    'var'       : 'float32',
    'std'       : 'float32',
    'confRate'  : 'float32',
    'nextclick' : 'int64',
    'mean'      : 'float32'
    }

parser = argparse.ArgumentParser(
    description='translate',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--debug', default=2, type=int, choices=[0,1,2])    

parser.add_argument('-d', '--dataset', type=str, default='train_translated',
    choices=['train_translated','test_translated','train_active_translated',
    'test_active_translated'])
# parser.add_argument('-d', '--dataset', type=str, default='train',choices=['train','test'])

def main():
    global args, DEBUG, DATASET
    args = parser.parse_args()
    DATASET = args.dataset
    DEBUG = args.debug
    print_debug(DEBUG)

    df = read_dataset(DATASET)
    if DEBUG:
        todir = '../processed_features_debug{}/'.format(DEBUG)
    else:
        todir = '../processed_features/'

    create_time(df, todir=todir, ext = '.pickle')
    # test()

def read_dataset(dataset):                   
    debug = DEBUG
    if debug:
        filename_train = '../input/debug{}/{}_textblob_debug{}.feather'.format(
                debug, 'train_translated', debug)  
        filename_test = '../input/debug{}/{}_textblob_debug{}.feather'.format(
                debug, 'test_translated', debug)                                            
    else:
        filename_train = '../input/{}_textblob.feather'.format('train_translated')  
        filename_test = '../input/{}_textblob.feather'.format('test_translated')  

    print_doing('reading train, test and merge')    
    df = read_train_test(filename_train, filename_test, '.feather', is_merged=1)
    print_memory()
    print(df.head())
    return df


MINH_LIST_NUNIQUE =[
    ['ip','mobile','day','hour'],
    ['ip','mobile_app','day','hour'],
    ['ip','mobile_channel','day','hour'],
    ['ip','app_channel','day','hour'],
    ['ip', 'mobile','app_channel','day','hour']
]

if __name__ == '__main__':
    main()