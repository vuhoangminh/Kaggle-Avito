   
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

def test_load_time():
    for name in ['train', 'test', 'train_active', 
        'test_active', 'periods_train', 'periods_test']:
        dstname = '../input/{}.feather'.format(name)
        t_start = time.time()
        print('>> loading...')
        df = mlc.load(dstname)
        t_end = time.time()
        print('loading time:', t_end-t_start)
        print('no. of rows:', len(df))
    

test_load_time()

