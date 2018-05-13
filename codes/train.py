import matplotlib
matplotlib.use('Agg')

import time
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
import nltk
# nltk.download('stopwords')


from lib.print_info import print_debug, print_doing, print_memory
from lib.read_write_file import save_csv, save_feather, save_file, save_pickle
from lib.read_write_file import load_csv, load_feather, load_pickle, read_train_test
from lib.prep_hdf5 import get_datatype, get_info_key_hdf5, add_dataset_to_hdf5
from lib.prepare_training import get_text_matrix, read_processed_h5, read_dataset
import lib.configs as configs
import features_list
from features_list import PREDICTORS

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

cwd = os.getcwd()
print ('working dir', cwd)
if 'codes' not in cwd:
    default_path = cwd + '/codes/'
    os.chdir(default_path)

parser = argparse.ArgumentParser(
    description='translate',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--debug', default=2, type=int, choices=[0,1,2])
parser.add_argument('-f', '--frac', default=1, type=float) 

SEED = 1988

CATEGORICAL = [
    'item_id', 'user_id', 'region', 'city', 'parent_category_name',
    'category_name', 'user_type', 'image_top_1', 'day', 'week', 'weekday',
    'cn_encoded', 'cty_encoded', 'img1_encoded', 'pcn_encoded',
    'reg_encoded', 'uid_encoded', 'uty_encoded',
]

REMOVED_LIST = [
    'user_id', 'region', 'city', 'parent_category_name',
    'category_name', 'user_type', 'image_top_1',
    'param_1', 'param_2', 'param_3', 'title', 'description',
    'activation_date', 'image'
]

TARGET = ['deal_probability']

def main():
    global args, DEBUG, FRAC, PREDICTORS
    args = parser.parse_args()
    DEBUG = args.debug
    FRAC = args.frac
    print_debug(DEBUG)

    print("\nData Load Stage")

    target = TARGET
    
    predictors = get_predictors()
    categorical = get_categorical(predictors)

    if DEBUG:
        mat_filename = '../processed_features_debug2/text_feature_kernel.pickle'
        dir_feature = '../processed_features_debug2/'
    else: 
        mat_filename = '../processed_features/text_feature_kernel.pickle'
        dir_feature = '../processed_features/'

    df, len_train, traindex, testdex = load_train_test()  
    df = drop_col(df,REMOVED_LIST)

    for feature in predictors:
        dir_feature_file = dir_feature + feature + '.pickle'
        if not os.path.exists(dir_feature_file):
            print('can not find {}. Please check'.format(dir_feature_file))
        else:
            if feature in df:
                print('{} already added'.format(feature))
            else:   
                print('\n>> adding {}'.format(feature))
                df = add_feature(df, dir_feature_file)
    
    print(df.info())
    print(df.head())
    print(df.tail())

def drop_col(df,cols):
    for col in cols:
        if col in df:
            df = df.drop([col], axis=1)
    return df

def load_train_test():
    train_df, test_df = read_dataset(False, DEBUG)
    len_train = len(train_df)

    train_df.set_index('item_id', inplace=True)
    traindex = train_df.index
    test_df.set_index('item_id', inplace=True)
    testdex = test_df.index
    
    y = train_df.deal_probability.copy()
    train_df.drop(TARGET, axis=1, inplace=True)

    print(train_df.info(), train_df.head())
    print(test_df.info(), test_df.head())  

    print('Train shape: {} Rows, {} Columns'.format(*train_df.shape))
    print('Test shape: {} Rows, {} Columns'.format(*test_df.shape))

    print("\nCombine Train and Test")
    df = pd.concat([train_df,test_df],axis=0)
    print(train_df.info())
    print(test_df.info())
    print(df.info())

    del train_df, test_df; gc.collect()
    print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))    

    return df, len_train, traindex, testdex

def add_time(df, filename):
    cols = ['day', 'week', 'weekday']
    gp = load_pickle(filename)
    for feature in cols:
        df[feature] = gp[feature].values
    return df

def add_feature(df, filename):
    gp = load_pickle(filename)
    for feature in gp:
        df[feature] = gp[feature].values
    return df    

def get_predictors():
    predictors = PREDICTORS
    print('------------------------------------------------')
    print('features:')
    for feature in predictors:
        print (feature)
    print('-- number of predictors:', len(predictors))        
    return predictors

def get_categorical(predictors):
    categorical = []
    for feature in predictors:
        if feature in CATEGORICAL:
            categorical.append(feature)
    print('------------------------------------------------')
    print('categorical:')
    for feature in categorical:
        print (feature)
    print('-- number of categorical features:', len(categorical))                        
    return categorical  

if __name__ == '__main__':
    main()