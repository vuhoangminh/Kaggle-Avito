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
parser.add_argument('-d', '--dataset', type=str, default='test_active',choices=['train_active','test_active'])
parser.add_argument('-f', '--fromiloc',default=0, type=int)
parser.add_argument('-b', '--debug',default=0, type=int)
parser.add_argument('-s', '--stepiloc',default=4000, type=int)
parser.add_argument('-rm', '--readmode',default='.pickle', type=str, choices=['.csv', '.feather', '.pickle'])
parser.add_argument('-cm', '--checkmode',default=0, type=int)

process = psutil.Process(os.getpid())

CAT_TRANSLATE = ['title']

def write_bash_missing_file(list_missing):
    str = ''
    for file in list_missing:
        temp = 'python -u build_dict_fast.py -f {} -s {} & '.format(file, 1)
        str = str + temp
    file = open('bash.txt','w') 
    file.write(str) 
    print (str)    

def main():
    global args, debug, NCHUNKS, STEP, READMODE, CHECKMODE
    args = parser.parse_args()
    which_dataset = args.dataset
    FROM = args.fromiloc
    debug = args.debug
    STEP = args.stepiloc
    READMODE = args.readmode
    CHECKMODE = args.checkmode

    if debug==2:
        NCHUNKS = 13
    if debug==1:
        NCHUNKS = 130    
    if debug==0:
        NCHUNKS = 2000  

    print('summary: debug {}, chunks {}, from {}'.format(debug, NCHUNKS, FROM))
    
    filename = '../input/' + which_dataset + '_title_unique' + READMODE
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
        train_df = pd.read_csv(filename, nrows=10, usecols=CAT_TRANSLATE)  
    if debug==1:
        train_df = pd.read_csv(filename, nrows=10000, usecols=CAT_TRANSLATE)
    if debug==0:
        train_df = pd.read_csv(filename, usecols=CAT_TRANSLATE)          
    return train_df  

def split_text_into_sentences(txt, N):
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
        if i%20==0:
            print(i) 
        if i%threshold==0: 
            print('{}/{}'.format(i,len(k_unique_element)))
        if not k_unique_element_null[i]:
            element_clean = ''.join(c for c in element if c <= '\uFFFF')
            sentences = split_text_into_sentences(element_clean, 1000)
            element_translated = ''
            for sentence in sentences:
                if debug: print('from:', sentence)
                translator = Translator()
                if debug: print(len(sentence))
                sentence_translated = translator.translate(sentence, dest='En')
                element_translated = element_translated + ' ' + sentence_translated.text
            map[element] = element_translated
            if debug: 
                print('from:', element)
                print('to:', element_translated)
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

def build_dict(unique_element, threshold, which_dataset, from_iloc):   
    range_split = range(0,len(unique_element),NCHUNKS)
    num_split = len(range(0,len(unique_element),NCHUNKS))
    print_memory()
    count_missing = 0
    list_missing = []
    for k in range(num_split):
        if k>=from_iloc and k<from_iloc+STEP:
            print('process {}/{}'.format(k,num_split-1))
            savename = '../dict_part/translated_{}_{}_{}.pickle'.format(which_dataset, 'title', k)
            if os.path.exists(savename):
                print('done already')
            else: 
                if CHECKMODE:
                    count_missing = count_missing + 1
                    list_missing.append(k)
                    print('missing', savename)
                else:
                    translate_and_save_each_chunk(k, num_split, 
                            range_split, unique_element, savename, threshold)               
    if CHECKMODE:
        print (count_missing, 'is missing')
        write_bash_missing_file(list_missing)
        # print (list_missing)



def translate_col_and_save(unique_element, which_dataset, from_iloc):
    if debug==2:
        build_dict(unique_element, 10, which_dataset, from_iloc)
    else:                           
        build_dict(unique_element, 100, which_dataset, from_iloc)   
                         
def drop_to_save_memory(df):
    for feature in df:
        if feature not in CAT_TRANSLATE:
            df = df.drop([feature], axis=1)
    return df            

def read_and_build_dict(filename, which_dataset, from_iloc):
    print('>> reading', filename)
    unique_element = pickle.load(open(filename, "rb" ))
    print_memory()
    translate_col_and_save(unique_element, which_dataset, from_iloc)

if __name__ == '__main__':
    main()