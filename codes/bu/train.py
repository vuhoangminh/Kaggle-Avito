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
from lib.prepare_training import get_text_matrix, read_processed_h5, read_dataset, drop_col, load_train_test, add_feature
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
    
    tabular_predictors = get_tabular_predictors()
    # categorical = get_categorical(predictors)

    if DEBUG:
        mat_filename = '../processed_features_debug2/text_feature_kernel.pickle'
        dir_feature = '../processed_features_debug2/'
    else: 
        mat_filename = '../processed_features/text_feature_kernel.pickle'
        dir_feature = '../processed_features/'

    X, y, test, full_predictors, predictors = prepare_training(mat_filename, dir_feature, tabular_predictors)

    categorical = get_categorical(predictors)
    predictors = get_predictors(predictors)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=SEED)

    print(X.shape)

    print("Light Gradient Boosting Regressor")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 15,
        # 'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        'learning_rate': 0.019,
        'verbose': 0
    }  

    print(lgbm_params)

    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X_train, y_train,
                    feature_name=full_predictors,
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                    feature_name=full_predictors,
                    categorical_feature = categorical)

    # Go Go Go
    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=16000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=200,
        verbose_eval=200
    )

def prepare_training(mat_filename, dir_feature, predictors):
    df, train_labels, len_train, traindex, testdex = load_train_test(['item_id'], TARGET, DEBUG)
    df = drop_col(df,REMOVED_LIST)

    # add features
    print_doing('---------------------------------------------------------------')
    print_doing('add tabular fetures')
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

    # add text_feature
    print_doing('---------------------------------------------------------------')
    print_doing('add text fetures')
    ready_df, tfvocab = get_text_matrix(mat_filename, 'all', 2, 0)

    # add text_feature
    print_doing('---------------------------------------------------------------')
    print_doing('stack')   
    X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
    testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])

    tfvocab = df.columns.tolist() + tfvocab
    for shape in [X,testing]:
        print("{} Rows and {} Cols".format(*shape.shape))
    print("Feature Names Length: ",len(tfvocab))
    # del df; gc.collect()  

    return X, train_labels, testing, tfvocab, df.columns.tolist()  

def get_tabular_predictors():
    predictors = PREDICTORS
    print('------------------------------------------------')
    print('load list:')
    for feature in predictors:
        print (feature)
    print('-- number of predictors:', len(predictors))        
    return predictors

def get_predictors(predictors):
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