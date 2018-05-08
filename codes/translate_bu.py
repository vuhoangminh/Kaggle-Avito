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
    
def map_translate(x, cols_translate):
    tqdm.pandas(tqdm())
    for feature in cols_translate:   
        print('>> translating', feature)
        x[feature]=x[feature].progress_map(translate)
        print('done translating', feature)
    return x

def exporter(x,dest_path):
    print("Writting to {}".format(dest_path))
    x.to_csv(dest_path,index=False)
    print("Done")

def read_file(filename):
    if debug==2:
        train_df = pd.read_csv(filename, nrows=10)  
    if debug==1:
        train_df = pd.read_csv(filename, nrows=1000)
    if debug==0:
        train_df = pd.read_csv(filename)          
    return train_df  

# CAT_TRANSLATE = ['parent_category_name', 'region', 'city',
#         'category_name', 'param_1', 'param_2', 'param_3', 
#         'title', 'description']

CAT_TRANSLATE = ['title', 'description']
# CAT_TRANSLATE = ['description']

def build_map(df, col, threshold):
    map = {np.nan:np.nan}
    unique_element = df[col].unique()
    print_memory()
    if debug: print (unique_element)

    unique_element_null = pd.isnull(unique_element)
    if debug: print('null:', unique_element_null)
    i = 0
    for element in unique_element:
        if i%threshold==0: 
            print('{}/{}'.format(i,len(unique_element)))
            translator = Translator()
            if i>0: 
                print('update map')
                map.update(map_temp)
            map_temp = dict()
        # translator = Translator()
        if debug: print ('doing', element)
        if not unique_element_null[i]:
            element_translated = translator.translate(element, dest='En')
            map[element] = element_translated.text
            if debug: print('to', element_translated.text)
        i = i+1            
    return map   

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, l, n))

import pickle

# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

from yandex import Translater
tr = Translater()
tr.set_key('trnsl.1.1.20180508T210255Z.2a05b4b2117bc805.9e960e745860b2c83026c719070f578e8d183226')
tr.set_from_lang('ru')
tr.set_to_lang('en')



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
                    print('len of 10 lement:', count_len)
                    count_len = 0
                if i%threshold==0: 
                    print('{}/{}'.format(i,len(k_unique_element)))
                    translator = Translator()
                if debug: print ('doing', element)
                if not k_unique_element_null[i]:
                    element_translated = translator.translate(element, dest='En')
                    count_len = count_len + len(element)
                    map[element] = element_translated.text
                    if debug: print('to', element_translated.text)
                i = i+1  
            print('saving to', savename)
            print_memory()
            with open(savename, 'wb') as handle:
                pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)


            del map; gc.collect()                    

    # return map   

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


def translate_col_to_en(df, col):
    if col=='title' or col=='description':
        if debug:
            map_dict = build_map(df, col, 10)
        else:                           
            map_dict = build_map(df, col, 200)   
    else:
        if debug:
            map_dict = build_map(df, col, 10)
        else:            
            map_dict = build_map(df, col, 400)                          
    if debug: print(map_dict)
    colname_translated = col
    print('mapping...')
    df[colname_translated] = df[col].apply(lambda x : map_dict[x])
    return df

def read_and_translate(filename, destname):
    print('>> reading...')
    df = read_file(filename)
    # df.head(5)

    for feature in df:
        df = desc_missing(df,feature)

    df_translated = df
    for feature in CAT_TRANSLATE:
        print('>> doing', feature)
        df_translated = translate_col_to_en(df_translated, feature)


    mlc.save(df_translated, destname) # DataFrames can be saved with ultra-fast feather format.
    del df_translated; gc.collect()
    df_translated = mlc.load(destname)
    print (df_translated.head())

def read_and_build_dict(filename, destname, which_dataset):
    print('>> reading...')
    df = read_file(filename)
    # df.head(5)

    for feature in df:
        df = desc_missing(df,feature)

    df_translated = df
    for feature in CAT_TRANSLATE:
        print('>> doing', feature)
        translate_col_and_save(df_translated, feature, which_dataset)



# read_and_translate(filename, destname)    

for dataset in ['train', 'test']:
# for dataset in ['test', 'train']:    
    filename = '../input/' + dataset + '.csv'
    destname = '../input/' + dataset + '_translated.feather'
    read_and_build_dict(filename, destname, dataset)
