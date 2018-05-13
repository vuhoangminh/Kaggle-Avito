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
import h5py

from lib.print_info import print_debug, print_doing, print_memory
from lib.read_write_file import save_csv, save_feather, save_file, save_pickle
from lib.read_write_file import load_csv, load_feather, load_pickle, read_train_test
from lib.prep_hdf5 import get_datatype, get_info_key_hdf5, add_dataset_to_hdf5, add_text_feature_to_hdf5
import lib.configs as configs
import features_list

SEED = configs.SEED
DATATYPE_DICT = configs.DATATYPE_DICT

cwd = os.getcwd()
print ('working dir', cwd)
if 'codes' not in cwd:
    default_path = cwd + '/codes/'
    os.chdir(default_path)

parser = argparse.ArgumentParser(
    description='translate',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--debug', default=2, type=int, choices=[0,1,2])    
# parser.add_argument('-d', '--dataset', type=str, default='train', choices=['train', 'test'])

def main():
    global args, DEBUG
    args = parser.parse_args()
    DEBUG = args.debug
    print_debug(DEBUG)
    for dataset in ['train', 'test']:
        do_dataset(dataset)
    write_all_feature_to_text()        

def do_dataset(dataset):
    train_df, test_df = read_dataset(False)
    len_train = len(train_df)

    if dataset=='train':
        df = train_df
        del test_df; gc.collect()
    else:
        df = test_df
        del train_df; gc.collect()    
    if DEBUG:
        storename = '../processed_features_debug{}/{}_debug{}.h5'.format(DEBUG, dataset, DEBUG)
        featuredir = '../processed_features_debug{}/'.format(DEBUG)
    else:
        storename = '../processed_features/{}.h5'.format(dataset)
        featuredir = '../processed_features/'

    add_dataset_to_hdf5(storename, df) 

    files = glob.glob(featuredir + '*.pickle') 
    for file in files:
        if 'text_feature_kernel' not in file:
            print(file)
            filename = file
            print ('\n>> doing', filename)
            df = load_pickle(filename)
            print_doing('extract')
            if DEBUG: print(df.head()); print(df.tail())
            if dataset=='train':
                df = df.iloc[:len_train]
                if DEBUG: print('train: ', df.head())
            else:
                df = df.iloc[len_train:]                        
                if DEBUG: print('test: ', df.tail())
            print('merging...')
            add_dataset_to_hdf5(storename, df)
            print_memory() 

def read_dataset(is_merged):                   
    debug = DEBUG
    if debug:
        filename_train = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'train', debug)  
        filename_test = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'test', debug)                                                                    
    else:
        filename_train = '../input/{}.feather'.format('train')  
        filename_test = '../input/{}.feather'.format('test')  

    print_doing('reading train, test and merge')  
    if is_merged:  
        df = read_train_test(filename_train, filename_test, '.feather', is_merged=True)
        if debug: print(df.head())
    else:
        train_df, test_df = read_train_test(filename_train, filename_test, '.feather', is_merged=False)        
        if debug: print(train_df.head()); print(test_df.head())
    print_memory()
    if is_merged:
        return df
    else:
        return train_df, test_df                

def read_dataset_original(is_merged):                   
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
    if is_merged:  
        df = read_train_test(filename_train, filename_test, '.feather', is_merged=True)
        if debug: print(df.head())
    else:
        train_df, test_df = read_train_test(filename_train, filename_test, '.feather', is_merged=False)        
        if debug: print(train_df.head()); print(test_df.head())
    print_memory()
    if is_merged:
        return df
    else:
        return train_df, test_df  



def write_all_feature_to_text():
    if DEBUG:
        storename = '../processed_features_debug{}/{}_debug{}.h5'.format(DEBUG, 'train', DEBUG)
    else:
        storename = '../processed_features/{}.h5'.format('train')
    with h5py.File(storename,'r') as hf:
        feature_list = list(hf.keys())
    filename = open('list_all_feature.txt', 'w')
    for item in feature_list:
        filename.write("%s\n" % item)

if __name__ == '__main__':
    main()
    # test()