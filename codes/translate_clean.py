debug = 0


if debug==2:
    NCHUNKS = 13
if debug==1:
    NCHUNKS = 130    
if debug==0:
    NCHUNKS = 2000            

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

process = psutil.Process(os.getpid())

def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', process.memory_info().rss/(2**30), ' GB')

def desc_missing(df, col):            
    if df[col].isnull().sum()>0:
        print("{} column has missing values, filling up with n/a".format(col))
        df[col].fillna("n/a",inplace=True)
        return df
    else:
        return df
    
def translate(x):
    try:
        return textblob.TextBlob(x).translate(to="en")
    except:
        return x


def read_file(filename):
    if debug==2:
        train_df = pd.read_csv(filename, nrows=10)  
    if debug==1:
        train_df = pd.read_csv(filename, nrows=1000)
    if debug==0:
        train_df = pd.read_csv(filename)          
    return train_df  


CAT_TRANSLATE = ['description']

def build_dict(df, col, threshold, which_dataset):
    unique_element = df[col].unique()
    if debug: print (unique_element)
    
    range_split = range(0,len(unique_element),NCHUNKS)
    num_split = len(range(0,len(unique_element),NCHUNKS))
    print_memory()
    for k in range(num_split):
        print('process {}/{}'.format(k+1,num_split))
        savename = 'translated_{}_{}_{}.pickle'.format(which_dataset, col, k)
        if os.path.exists(savename):
            print('done already')
        else:            
            map = {np.nan:np.nan}    
            if k+1<num_split:
                k_unique_element = unique_element[range_split[k]:range_split[k+1]]
            else:
                k_unique_element = unique_element[range_split[k]:]  
            if debug: print (k_unique_element)               
            print_memory()                

            k_unique_element_null = pd.isnull(k_unique_element)            
            if debug: print('null:', k_unique_element_null)
            i = 0
            count_len = 0
            for element in k_unique_element:
                if i%10==0: 
                    print(i)
                    # print('len of 10 lement:', count_len)
                    count_len = 0
                if i%threshold==0: 
                    print('{}/{}'.format(i,len(k_unique_element)))
                    translator = Translator()
                if debug: print ('doing', element)
                if not k_unique_element_null[i]:
                    element_translated = translator.translate(element, dest='En')
                    count_len = count_len + len(element)
                    if debug: print('to', element_translated)
                    map[element] = element_translated.text
                    if debug: print('to', element_translated.text)
                i = i+1  
            print('saving to', savename)
            print_memory()
            with open(savename, 'wb') as handle:
                pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)
            del map; gc.collect()                    

def translate_col_and_save(df, col, which_dataset):
    if col=='title':
        if debug==2:
            build_dict(df, col, 10, which_dataset)
        else:                           
            build_dict(df, col, 200, which_dataset)   
    elif col=='description':            
        if debug==2:
            build_dict(df, col, 10, which_dataset)
        else:                           
            build_dict(df, col, 1, which_dataset)           
    else:
        if debug==2:
            build_dict(df, col, 10, which_dataset)
        else:            
            build_dict(df, col, 400, which_dataset)                          

def read_and_build_dict(filename, destname, which_dataset):
    print('>> reading', which_dataset)
    df = read_file(filename)
    # df.head(5)

    for feature in df:
        df = desc_missing(df,feature)

    df_translated = df
    for feature in CAT_TRANSLATE:
        print('>> doing', feature)
        translate_col_and_save(df_translated, feature, which_dataset)

   
for dataset in ['train', 'test']: 
    filename = '../input/' + dataset + '.csv'
    destname = '../input/' + dataset + '_translated.feather'
    read_and_build_dict(filename, destname, dataset)
