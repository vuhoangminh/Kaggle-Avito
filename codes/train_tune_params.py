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


from lib.print_info import print_debug, print_doing, print_memory, print_doing_in_task, print_header
from lib.read_write_file import save_csv, save_feather, save_file, save_pickle
from lib.read_write_file import load_csv, load_feather, load_pickle, read_train_test
from lib.prep_hdf5 import get_datatype, get_info_key_hdf5, add_dataset_to_hdf5
from lib.prepare_training import get_text_matrix, read_processed_h5, read_dataset, drop_col, load_train_test, add_feature, get_string_time
import lib.configs as configs
import features_list
from features_list import PREDICTORS_BASED
from train import predict_sub, train, prepare_training, get_tabular_predictors, get_categorical, get_predictors, get_good_local

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

SEED = 1988
yearmonthdate_string = get_string_time()
np.random.seed(SEED)

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
parser.add_argument('-tm', '--trainmode', default='gbdt', type=str, choices=['gbdt', 'dart']) 
parser.add_argument('-o', '--option', default=0, type=int) 

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
    global args, DEBUG, FRAC, PREDICTORS, TRAINMODE, OPTION
    args = parser.parse_args()
    DEBUG = args.debug
    FRAC = args.frac
    TRAINMODE = args.trainmode
    OPTION=args.option
    print_debug(DEBUG)
    feature_train = get_good_local()
    if DEBUG: print(feature_train)
    PREDICTORS = PREDICTORS_BASED + feature_train
    if DEBUG: print(PREDICTORS)
    DO()       

def DO():
    tabular_predictors = get_tabular_predictors()
    if DEBUG:
        mat_filename = '../processed_features_debug2/text_feature_kernel.pickle'
        dir_feature = '../processed_features_debug2/'
    else: 
        mat_filename = '../processed_features/text_feature_kernel_30000.pickle'
        dir_feature = '../processed_features/'
    X, y, test, full_predictors, predictors, testdex = prepare_training(mat_filename, dir_feature, 
            tabular_predictors, is_textadded=True)
    categorical = get_categorical(predictors)
    predictors = get_predictors(predictors)

    # num_leave_list = [7,16,32]
    num_leave_list = [-1]
    if TRAINMODE == 'gbdt':
        boosting_list = ['gbdt']
    else:
        boosting_list = ['dart']

    for boosting_type in boosting_list:
        for num_leave in num_leave_list:
            if DEBUG:
                subfilename = '../sub/debug_{}_{}_{}features_numleave{}_maxdepth15_OPTION{}.csv.gz'. \
                        format(yearmonthdate_string,boosting_type,str(len(predictors)),num_leave,OPTION)
            else:
                subfilename = '../sub/{}_{}_{}features_numleave{}_maxdepth15_OPTION{}.csv.gz'. \
                        format(yearmonthdate_string,boosting_type,str(len(predictors)),num_leave,OPTION)                        
            if os.path.exists(subfilename) and not DEBUG:
                print('{} done already'.format(subfilename))     
            else:                               
                model_lgb, subfilename = train(X,y,num_leave,full_predictors,
                        categorical,predictors,boosting_type,option=OPTION)
                predict_sub(model_lgb, testdex, test, subfilename)
                del model_lgb; gc.collect()

if __name__ == '__main__':
    main()