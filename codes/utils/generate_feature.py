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
import psutil
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import time
import nltk, textwrap

process = psutil.Process(os.getpid())

def main():
    global args, debug, NCHUNKS
    args = parser.parse_args()
    datasets = args.dataset
    debug = args.debug

    if debug==2:
        NCHUNKS = 13
    if debug==1:
        NCHUNKS = 130    
    if debug==0:
        NCHUNKS = 2000  

    print('summary: debug {}, chunks {}'.format(debug, NCHUNKS))
    
    if datasets == 'all':
        datasets = ['train', 'test']
    else:
        datasets = [datasets]                

    for which_dataset in datasets:        
        filename = '../input/' + which_dataset + '.csv'
        read_and_build_map(filename, which_dataset)
        df_translated = read_and_translate(filename, which_dataset)
        destname = '../input/' + which_dataset + '.feather'
        print('>> saving to ...', destname)
        mlc.save(df_translated, destname) # DataFrames can be saved with ultra-fast feather format.
        del df_translated; gc.collect()
        print('>> loading ...', destname)
        df_translated = mlc.load(destname)
        print (df_translated.head())
        df2 = df_translated.sample(frac=0.01)
        print(df2.head(5))
        print(df2.tail(5))


def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', round(process.memory_info().rss/(2**30),2), ' GB')

def desc_missing(df, col):            
    if df[col].isnull().sum()>0:
        print("{} column has missing values, filling up with n/a".format(col))
        df[col].fillna("n/a",inplace=True)
        return df
    else:
        return df
    
def read_file(filename):
    if debug==2:
        train_df = pd.read_csv(filename, nrows=10)  
    if debug==1:
        train_df = pd.read_csv(filename, nrows=10000)
    if debug==0:
        train_df = pd.read_csv(filename)          
    return train_df  

def read_and_translate(filename, which_dataset):
    print('>> reading', which_dataset)
    df = read_file(filename)
    for feature in df:
        df = desc_missing(df,feature)
    df_translated = df
    for feature in CAT_TRANSLATE:
        dstname = '../dict/dict_ru_to_en_{}_{}.pickle'.format(which_dataset, feature)
        map_dict = pickle.load(open(dstname, "rb" ))
        new_feature = feature + '_en'
        df_translated[new_feature] = df[feature].apply(lambda x : map_dict[x])        
                                
    return df_translated 

def read_and_build_map(filename, which_dataset):
    print('>> reading', which_dataset)
    df = read_file(filename)
    # df.head(5)

    for feature in df:
        df = desc_missing(df,feature)

    for feature in CAT_TRANSLATE:
        map_dict = dict()
        dstname = '../dict/dict_ru_to_en_{}_{}.pickle'.format(which_dataset, feature)
        print('>> doing', feature)
        unique_element = df[feature].unique()
        num_split = len(range(0,len(unique_element),NCHUNKS))
        print_memory()
        is_cont = True
        if os.path.exists(dstname):
            print('done already')
        else:            
            for k in range(num_split):
                if is_cont:
                    savename = '../dict/translated_{}_{}_{}.pickle'.format(which_dataset, feature, k)
                    if os.path.exists(savename):
                        print('loading', savename)
                        map_temp = pickle.load(open(savename, "rb" ))
                        print('updating map')
                        map_dict.update(map_temp)
                    else:
                        print('missing', savename, '. Please check!!')
                        is_cont = False
            if is_cont:
                print('saving final dict to', dstname)
                with open(dstname, 'wb') as handle:
                    pickle.dump(map_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)                 

if __name__ == '__main__':
    main()