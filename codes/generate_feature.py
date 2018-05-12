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
from lib.gen_feature import generate_groupby_by_type_and_columns, create_time, measure_length, map_key, create_text_feature
import lib.configs as configs
import features_list

SEED = configs.SEED
DATATYPE_DICT = configs.DATATYPE_DICT
NAMEMAP_DICT = configs.NAMEMAP_DICT
MINH_LIST_MEAN_DEAL_PROB = features_list.MINH_LIST_MEAN_DEAL_PROB

cwd = os.getcwd()
print ('working dir', cwd)
if 'codes' not in cwd:
    default_path = cwd + '/codes/'
    os.chdir(default_path)

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

    gen_time_feature(df, todir, '.pickle')
    gen_len_title_description_feature(df, todir, '.pickle')
    gen_mean_deal_probability (df, todir, '.pickle')
    gen_text_feature_from_kernel (df, todir, '.pickle', 'russian')

    ## after translated!!
    gen_text_feature_from_kernel (df, todir, '.pickle', 'english')

def gen_text_feature_from_kernel(df, todir, ext, language):
    create_text_feature (df, todir, ext, language)
    # if DEBUG: print(df['activation_date'].head()), print (gp.head())
    # del gp; gc.collect()
    # print_memory()    
    
def gen_time_feature(df, todir, ext):
    gp = create_time(df, todir=todir, ext = ext)
    if DEBUG: print(df['activation_date'].head()), print (gp.head())
    del gp; gc.collect()
    print_memory()

def gen_len_title_description_feature(df, todir, ext):
    selcols = ['title_en','description_en','title','description']
    gp = measure_length(df, selcols=selcols, todir=todir, ext = '.pickle')
    if DEBUG: print(df[selcols].head()), print (gp.head())
    del gp; gc.collect()    
    print_memory()

def gen_mean_deal_probability (df, todir, ext):
    for selcols in MINH_LIST_MEAN_DEAL_PROB:
        gp = generate_groupby_by_type_and_columns(df, selcols, 'mean', todir, ext)
        if DEBUG: print(df[selcols].head()), print (gp.head())
        del gp; gc.collect()    
        print_memory()

def read_dataset(dataset):                   
    debug = DEBUG
    if debug:
        filename_train = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'train_translated', debug)  
        filename_test = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'test_translated', debug)                                                                    
    else:
        filename_train = '../input/{}.feather'.format('train_translated')  
        filename_test = '../input/{}.feather'.format('test_translated')  

    print_doing('reading train, test and merge')    
    df = read_train_test(filename_train, filename_test, '.feather', is_merged=1)
    print_memory()
    print(df.head())
    return df

def test():
    print (map_key('user_id'))
    print (map_key('abc'))

if __name__ == '__main__':
    main()
    # test()