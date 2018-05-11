import pickle
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


SEED = 1988


# train               1503424
# test                508438
# train_active        14129821
# test_active         12824068
# periods_train       16687412
# periods_test        13724922


def convert_to_feather(filename, dstname):
    print('>> reading', filename)
    df = pd.read_csv(filename)   
    print('no. of rows of filename:', len(df))
    print('>> saving to ...', dstname)
    mlc.save(df, dstname)

def convert():    
    for name in ['train', 'test', 'train_active', 
            'test_active', 'periods_train', 'periods_test']:
          filename = '../input/{}.csv'.format(name)
          dstname = '../input/{}.feather'.format(name)
          convert_to_feather(filename, dstname)

def write_debug_mode_csv():
    for name in ['train', 'test', 'train_active',  'test_active', 
                    'periods_train', 'periods_test', 'train_translated', 
                    'train_active_translated', 'test_translated', 'test_active_translated']:
          for debug in [1,2]:
                    print('----------------------------------------------------------------')
                    dstname = '../input/{}.feather'.format(name)
                    t_start = time.time()
                    print('>> loading', dstname)
                    df = mlc.load(dstname)
                    print('no. of rows:', len(df))
                    print(df.head())
                    # del df; gc.collect()
                    t_end = time.time()
                    print('loading time:', t_end-t_start)

                    savename = '../input/debug{}/{}_debug{}.csv'.format(debug,name,debug)
                    if debug == 1:
                              df_extracted = df.sample(frac=0.1, random_state = SEED)
                    else:
                              df_extracted = df.sample(frac=0.001, random_state = SEED) 
                    print('no. of rows:', len(df_extracted))        
                    print('>> saving to', savename)
                    df_extracted.to_csv(savename,index=False)    
                    print('done')                                                                                                                                                           

def convert_debug_mode_feather():
    for name in ['train', 'test', 'train_active',  'test_active', 
                    'periods_train', 'periods_test', 'train_translated', 
                    'train_active_translated', 'test_translated', 'test_active_translated']:
          for debug in [1,2]:
                    print('----------------------------------------------------------------')
                    dstname = '../input/debug{}/{}_debug{}.csv'.format(debug,name,debug)
                    t_start = time.time()
                    print('>> loading', dstname)
                    df = pd.read_csv(dstname)
                    print('no. of rows:', len(df))
                    print(df.head())
                    t_end = time.time()
                    print('loading time:', t_end-t_start)

                    savename = '../input/debug{}/{}_debug{}.feather'.format(debug,name,debug)
                    print('>> saving to', savename)
                    mlc.save(df, savename)   
                    print('done')  

# test_load_time()
# write_debug_mode_csv()
convert_debug_mode_feather()


