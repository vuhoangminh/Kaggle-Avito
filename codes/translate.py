
# coding: utf-8

# In[1]:


debug = 0


import pandas as pd
import sys
import textblob
from tqdm import tqdm,tqdm_pandas
import math
import mlcrate as mlc
import gc
import numpy as np
from googletrans import Translator

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



CAT_TRANSLATE = ['parent_category_name', 'region', 'city',
        'category_name', 'param_1', 'param_2', 'param_3', 
        'title', 'description']

# CAT_TRANSLATE = ['param_3', 
#         'title', 'description',
#         'region', 'city']


from transliterate import translit, get_available_language_codes

def build_map(df, col, threshold):
    map = {np.nan:np.nan}
    unique_element = df[col].unique()
    if debug: print (unique_element)

    unique_element_null = pd.isnull(unique_element)
    if debug: print('null:', unique_element_null)
    i = 0
    for element in unique_element:
        if i%threshold==0: 
            print('{}/{}'.format(i,len(unique_element)))
            translator = Translator()
        # translator = Translator()
        if debug: print ('doing', element)
        if not unique_element_null[i]:
            element_translated = translator.translate(element, dest='En')
            map[element] = element_translated.text
            if debug: print('to', element_translated.text)
        i = i+1            
    return map   

def translate_col_to_en(df, col):
    if col=='title' or col=='description':
        map_dict = build_map(df, col, 200)   
    else:
        map_dict = build_map(df, col, 400)                    
    if debug: print(map_dict)
    colname_translated = col
    print('mapping...')
    df[colname_translated] = df[col].apply(lambda x : map_dict[x])
    return df

FILE = 'test'
filename = '../input/' + FILE + '.csv'
destname = '../input/' + FILE + '_translated.feather'
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

read_and_translate(filename, destname)    