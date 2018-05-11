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
    string = '()'

if __name__ == '__main__':
    main()
