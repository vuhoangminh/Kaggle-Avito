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
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import time
import nltk, textwrap

from lib.print_info import print_debug, print_memory
from lib.read_file import load_feather, load_csv, load_pickle, read_train_test

SEED = 1988

DATATYPE_DICT = {
    'count'     : 'uint32',
    'nunique'   : 'uint32',
    'cumcount'  : 'uint32',
    'var'       : 'float32',
    'std'       : 'float32',
    'confRate'  : 'float32',
    'nextclick' : 'int64',
    'mean'      : 'float32'
    }

parser = argparse.ArgumentParser(
    description='translate',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--debug', default=2, type=int, choices=[0,1,2])    
parser.add_argument('-d', '--dataset', type=str, default='train',choices=['train','test'])

def main():
    global args, DEBUG, DATASET
    args = parser.parse_args()
    DATASET = args.dataset
    DEBUG = args.debug
    print_debug(DEBUG)
    test()

def test():                   
    debug = DEBUG
    if debug==3:
        debug=2
    name = DATASET
    if debug:
        savename = '../input/debug{}/{}_textblob_debug{}.feather'.format(
                debug, name, debug)                    
    else:
        dstname = '../input/{}.feather'.format(name)
        savename = '../input/{}_textblob.feather'.format(name)                                                                                                                                                          
 
if __name__ == '__main__':
    main()