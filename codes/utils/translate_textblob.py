import pickle
import argparse
import pandas as pd
import sys, os
import textblob
from tqdm import tqdm, tqdm_pandas
import math
import mlcrate as mlc
import gc
import numpy as np
from googletrans import Translator
import pickle
import psutil
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import time
import nltk, textwrap
import pandas as pd
import sys
import textblob
from tqdm import tqdm, tqdm_pandas

process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser(
    description='translate',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--debug', default=0, type=int)
parser.add_argument('-d', '--dataset', type=str, default='train_translated',
        choices=['train_translated', 'train_active_translated', 'test_translated',
        'test_active_translated'])

CAT_TRANSLATE = ['title']

def main():
    global args, DEBUG, DATASET
    args = parser.parse_args()
    DATASET = args.dataset
    DEBUG = args.debug
    print('summary: debug {}, dataset {}'.format(DEBUG, DATASET))
    translate_textblob()

def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', round(process.memory_info().rss/(2**30),2), ' GB')

def translate_textblob():
    print(
            '--------------------------------------------------------------------------'
    )
    debug = DEBUG
    if debug==3:
        debug=2
    name = DATASET
    if debug:
            dstname = '../input/debug{}/{}_debug{}.feather'.format(
                    debug, name, debug)
            savename = '../input/debug{}/{}_textblob_debug{}.feather'.format(
                    debug, name, debug)                    
    else:
            dstname = '../input/{}.feather'.format(name)
            savename = '../input/{}_textblob.feather'.format(name) 


    if os.path.exists(savename):
        print('done already')
    else:                    
        t_start = time.time()
        print('>> loading', dstname)
        df = mlc.load(dstname)
        if DEBUG == 3:
            df = df.sample(frac=0.01)
        print('no. of rows:', len(df))
        t_end = time.time()
        print('loading time:', t_end - t_start)
        print_memory()

        print('>> translating')
        df_translated = map_translate(df)
        print (df_translated.head())
        print (df_translated.tail())

        print('>> saving', savename)
        mlc.save(df_translated, savename)


def translate(x):
    try:
        return textblob.TextBlob(x).translate(from_lang='ru', to="en")
    except:
        return x

def map_translate(x):
    print("Begining to translate")
    tqdm.pandas(tqdm())
    print("Begining to translate Title")
    x['title_en_textblob'] = x['title'].progress_map(translate)
    x['title_en_textblob'] = x['title_en_textblob'].astype('str') 
    print("Done translating")
    return x

if __name__ == '__main__':
    main()
