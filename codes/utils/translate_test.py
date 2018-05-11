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

# import codecs

parser = argparse.ArgumentParser(
    description='translate',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset', type=str, default='train',choices=['train','test'])
parser.add_argument('-f', '--fromiloc',default=0, type=int)
parser.add_argument('-b', '--debug',default=0, type=int)
parser.add_argument('-s', '--stepiloc',default=1000, type=int)

process = psutil.Process(os.getpid())

CAT_TRANSLATE = ['description']
# CAT_TRANSLATE = ['title']

def main():
    global args, debug, NCHUNKS,STEP
    args = parser.parse_args()
    which_dataset = args.dataset
    FROM = args.fromiloc
    debug = args.debug
    STEP = args.stepiloc

    if debug==2:
        NCHUNKS = 13
    if debug==1:
        NCHUNKS = 130    
    if debug==0:
        NCHUNKS = 2000  

    print('summary: debug {}, chunks {}, from {}'.format(debug, NCHUNKS, FROM))
    
    filename = '../input/' + which_dataset + '.csv'
    read_and_build_dict(filename, which_dataset, FROM)

def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', round(process.memory_info().rss/(2**30),2), ' GB')

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
        train_df = pd.read_csv(filename, nrows=10000)
    if debug==0:
        train_df = pd.read_csv(filename)          
    return train_df  

# CAT_TRANSLATE = ['parent_category_name', 'region', 'city',
#         'category_name', 'param_1', 'param_2', 'param_3', 
#         'title', 'description']

# CAT_TRANSLATE = ['title', 'description']
def split_text_into_sentences(txt, N):
    # tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    # sentences = tokenizer.tokenize(txt)
    sentences = textwrap.wrap(txt, N, break_long_words=False)
    return sentences        


def translate_and_save_each_chunk_description(k, num_split, range_split, unique_element, 
            savename, threshold):
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
    for element in k_unique_element:
        if i%threshold==0: 
            print('{}/{}'.format(i,len(k_unique_element)))
        if not k_unique_element_null[i]:
            # sentences = split_text_into_sentences(element, 1000)
            # element_translated = ''
            # for sentence in sentences:
            #     if debug: print('from:', sentence)
            #     translator = Translator()
            #     print(len(sentence))
            #     sentence_translated = translator.translate(sentence, dest='En')
            #     element_translated = element_translated + ' ' + sentence_translated.text
            translator = Translator()
            element_clean = ''.join(c for c in element if c <= '\uFFFF')
            if debug: print('from', element)
            element_translated = translator.translate(element_clean, dest='En')
            if debug: print('to', element_translated.text)    
            map[element] = element_translated.text
                    

        i = i+1 
    if not debug:         
        print('saving to', savename)
        with open(savename, 'wb') as handle:
            pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print_memory()        
    del map; gc.collect()  

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

def translate_and_save_each_chunk(k, num_split, range_split, unique_element, 
            savename, threshold):
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
    if not debug:         
        print('saving to', savename)
        with open(savename, 'wb') as handle:
            pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print_memory()        
    del map; gc.collect()      

def build_dict(df, col, threshold, which_dataset, from_iloc):
    unique_element = df[col].unique()
    # if debug: print (unique_element.encode("utf-8"))
    
    range_split = range(0,len(unique_element),NCHUNKS)
    num_split = len(range(0,len(unique_element),NCHUNKS))
    print_memory()
    for k in range(num_split):
        if k>=from_iloc and k<from_iloc+STEP:
            print('process {}/{}'.format(k,num_split-1))
            savename = '../dict/translated_{}_{}_{}.pickle'.format(which_dataset, col, k)
            if os.path.exists(savename):
                print('done already')
            else:     
                if col!='description':           
                    translate_and_save_each_chunk(k, num_split, 
                            range_split, unique_element, savename, threshold)
                else:
                    translate_and_save_each_chunk_description(k, num_split, 
                            range_split, unique_element, savename, threshold)                  

    # return map   

def translate_col_and_save(df, col, which_dataset, from_iloc):
    if col=='title':
        if debug==2:
            build_dict(df, col, 10, which_dataset, from_iloc)
        else:                           
            build_dict(df, col, 200, which_dataset, from_iloc)   
    elif col=='description':            
        if debug==2:
            build_dict(df, col, 10, which_dataset, from_iloc)
        else:                           
            build_dict(df, col, 1, which_dataset, from_iloc)           
    else:
        if debug==2:
            build_dict(df, col, 10, which_dataset, from_iloc)
        else:            
            build_dict(df, col, 400, which_dataset, from_iloc)                          


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

def read_and_build_dict(filename, which_dataset, from_iloc):
    print('>> reading', which_dataset)
    df = read_file(filename)
    # df.head(5)

    for feature in df:
        df = desc_missing(df,feature)

    df_translated = df
    for feature in CAT_TRANSLATE:
        print('>> doing', feature)
        translate_col_and_save(df_translated, feature, which_dataset, from_iloc)

def count_greater_N_character(df, col, N):
    map = dict()
    count = 0
    max_len = 0
    min_len = 100000000
    print('>> counting')
    for index, row in df.iterrows():
        e = row[col]
        len_e = len(e)
        if max_len<len_e: 
            max_len = len_e
        if min_len>len_e: 
            min_len = len_e            
        # if len_e>N: print(index, len_e)
        if len_e>N: 
            count=count+1
            map[e] = len_e
    return map, count, max_len, min_len

def read_and_count_character(filename, which_dataset, from_iloc):
    print('>> reading', which_dataset)
    df = read_file(filename)
    
    for feature in df:
        df = desc_missing(df,feature)

    map, count, max_len, min_len = count_greater_N_character(df, 'description', 3000)
    print(count, 'elements')
    print('max', max_len)
    print('min', min_len)
    # print(map)


if __name__ == '__main__':
    main()