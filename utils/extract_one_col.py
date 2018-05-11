   
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

def reduce():    
    for name in ['train_active', 
            'test_active']:
        filename = '../input/{}.csv'.format(name)
        df = pd.read_csv(filename, usecols=['title'])  
        dstname =  '../input/{}_title.feather'.format(name)
        mlc.save(df, dstname)
        savename = '../input/{}_title_unique.pickle'.format(name)
        unique_element = df['title'].unique()
        with open(savename, 'wb') as handle:
            pickle.dump(unique_element, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
reduce()

