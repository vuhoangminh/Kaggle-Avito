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

parser = argparse.ArgumentParser(
    description='translate',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--debug',default=0, type=int)


def main():
    global args, DEBUG
    args = parser.parse_args()
    DEBUG = args.debug
    check_if_all_translated()

CAT_TRANSLATE = ['parent_category_name', 'region', 'city',
        'category_name', 'param_1', 'param_3', 
        'title', 'description']

# CAT_TRANSLATE = ['parent_category_name', 'param_2']

def find_unique_element_and_count(list_not_translated):
    output = set()
    for x in list_not_translated:
        output.add(x)
    return output, len(output)

def check_if_all_translated():
    for name in ['train_translated', 'train_active_translated', 
                'test_translated', 'test_active_translated']:     
        print('--------------------------------------------------------------------------')
        debug = DEBUG
        if debug:
            dstname = '../input/debug{}/{}_debug{}.feather'.format(debug,name,debug)
        else:            
            dstname = '../input/{}.feather'.format(name)
        t_start = time.time()
        print('>> loading', dstname)
        df = mlc.load(dstname)
        print('no. of rows:', len(df))
        t_end = time.time()
        print('loading time:', t_end-t_start)
        for feature in CAT_TRANSLATE:
            print('>> doing', feature)
            list_not_translated = []
            count_not_translated = 0
            for index, row in df.iterrows():
                if index%100000==0: print(index,'/',len(df))
                if row[feature] == row[feature + '_en']:
                    count_not_translated = count_not_translated + 1
                    list_not_translated = list_not_translated + [row[feature]]
            
            print('feature {} not translated {}/{}'.format(feature,count_not_translated,len(df)))  
            list_not_translated_unique, count_not_translated_unique = find_unique_element_and_count(list_not_translated)                             
            print('list not translated', list_not_translated_unique)
            print('count not translated', count_not_translated_unique)
                

if __name__ == '__main__':
    main()