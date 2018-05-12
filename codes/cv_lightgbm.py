import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import argparse
import sys, os
import textblob
import pandas as pd
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime
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
import h5py
from scipy.sparse import hstack, csr_matrix

from lib.print_info import print_debug, print_doing, print_memory
from lib.read_write_file import save_csv, save_feather, save_file, save_pickle
from lib.read_write_file import load_csv, load_feather, load_pickle, read_train_test
from lib.prep_hdf5 import get_datatype, get_info_key_hdf5, add_dataset_to_hdf5
from lib.prepare_training import get_text_matrix, read_processed_h5
import lib.configs as configs
import features_list
from features_list import PREDICTORS

from sklearn import preprocessing

SEED = 1988

now = datetime.datetime.now()
if now.month<10:
    month_string = '0'+str(now.month)
else:
    month_string = str(now.month)
if now.day<10:
    day_string = '0'+str(now.day)
else:
    day_string = str(now.day)
yearmonthdate_string = str(now.year) + month_string + day_string

boosting_type = 'gbdt'
TARGET = ['deal_probability']
	
CATEGORICAL = [
    'item_id', 'user_id', 'region', 'city', 'parent_category_name',
    'category_name', 'param_1', 'param_2', 'param_3', 'title',
    'description', 'activation_date', 'user_type', 'image', 'image_top_1',
    'day', 'week', 'weekday'
]

REMOVED_LIST = [
    'title', 'description', 'param_1', 'param_2', 'param_3', 
    'price', 'image', 'image_top_1', 'activation_date', 'deal_probability',
    'uid_cn_mean_dp', 'uid_mean_dp', 'uid_pcn_cn_mean_dp', 'uid_pcn_mean_dp',
    'item_id', 'user_id'
]

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


def main():
    global args, DEBUG, FRAC, PREDICTORS
    args = parser.parse_args()
    DEBUG = args.debug
    FRAC = args.frac
    print_debug(DEBUG)

    if DEBUG:
        storename = '../processed_features_debug{}/{}_debug{}.h5'.format(DEBUG, 'train', DEBUG)
        mat_filename = '../processed_features_debug{}/text_feature_kernel.pickle'.format(DEBUG)
    else:
        storename = '../processed_features/{}.h5'.format('train')
        mat_filename = '../processed_features_debug{}/text_feature_kernel.pickle'.format(DEBUG)


    PREDICTORS = get_predictors(storename)
    
    DO(mat_filename, storename,7,3,1)


def get_predictors(storename):
    # with h5py.File(storename,'r') as hf:
    #     feature_list = list(hf.keys())
    # predictors = []
    # for feature in feature_list:
    #     if '_en' not in feature and feature not in REMOVED_LIST:
    #         predictors = predictors + [feature]
    return PREDICTORS


def get_categorical(predictors):
    categorical = []
    for feature in predictors:
        if feature in CATEGORICAL:
            categorical.append(feature)
    print('------------------------------------------------')
    print('categorical:')
    for feature in categorical:
        print (feature)
    print('number of categorical features:', len(categorical))                        
    return categorical  

def create_list(loop_count):
  return ''.join([num for num in range(loop_count)])

def DO(mat_filename, storename,num_leaves,max_depth, option):
    frac = FRAC
    print('------------------------------------------------')
    print('start...')
    print('fraction:', frac)
    print('prepare predictors, categorical and target...')
    predictors = PREDICTORS
    categorical = get_categorical(predictors)
    target = TARGET

    subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_cv2_' + str(int(100*frac)) + \
            'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(option) + '.csv.gz'
    modelfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
            'features_' + boosting_type + '_cv2_' + str(int(100*frac)) + \
            'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(option)

    print('----------------------------------------------------------')
    print('SUMMARY:')
    print('----------------------------------------------------------')
    print('predictors:',predictors)
    print('taget', target)
    print('categorical', categorical)
    print('submission file name:', subfilename)
    print('model file name:', modelfilename)
    print('fraction:', frac)
    print('option:', option)

    print('----------------------------------------------------------')
    train_df = read_processed_h5(storename, predictors+target, categorical)
    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        train_df[col] = lbl.fit_transform(train_df[col].astype(str))

    if DEBUG: print(train_df.head()); print(train_df.info())
    # train_df = train_df.sample(frac=frac, random_state = SEED)
    print_memory('afer reading train:')
    print(train_df.head())
    print("train size: ", len(train_df))
    gc.collect()

    print('----------------------------------------------------------')
    print("Training...")
    start_time = time.time()

    params = {
        'boosting_type': boosting_type,
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.1,
        'num_leaves': num_leaves,  # we should let it be smaller than 2^(max_depth)
        'max_depth': max_depth,  # -1 means no limit
        'subsample': 0.9,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'feature_fraction': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 10,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0
    }
    print ('params:', params)

    print_doing('cleaning train...')
    train_df_array = train_df[predictors].values
    train_df_labels = train_df[target].values.astype('int').flatten()
    del train_df; gc.collect()
    print_memory()

    print_doing('reading text matrix')
    train_mat_text = get_text_matrix(mat_filename, 'train', DEBUG, train_df_array.shape[0])

    print_doing('stack two matrix')
    train_df_array = hstack([csr_matrix(train_df_array),train_mat_text])

    # for i in range(train_mat_text.shape[1]):

    lst = list(range(train_mat_text.shape[1]))
    new_predictors = [str(i) for i in lst]
    
    
    # create_list(train_mat_text.shape[1])
    predictors = predictors + new_predictors

    del train_mat_text; gc.collect()

    print('>> prepare dataset...')
    dtrain_lgb = lgb.Dataset(train_df_array, label=train_df_labels,
                        feature_name=predictors,
                        categorical_feature=categorical)
    del train_df_array, train_df_labels; gc.collect()                        
    print_memory()                        
    
    print('>> start cv...')

    cv_results  = lgb.cv(params, 
                        dtrain_lgb, 
                        categorical_feature = categorical,
                        num_boost_round=20000,                       
                        metrics='rmse',
                        seed = SEED,
                        shuffle = False,
                        # stratified=True, 
                        nfold=10, 
                        show_stdv=True,
                        early_stopping_rounds=100, 
                        verbose_eval=True)                     


    print('[{}]: model training time'.format(time.time() - start_time))
    print_memory()


    # print (cv_results)
    print('--------------------------------------------------------------------') 
    num_boost_rounds_lgb = len(cv_results['rmse-mean'])
    print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))

    print ('>> start trainning... ')
    model_lgb = lgb.train(
                        params, dtrain_lgb, 
                        num_boost_round=num_boost_rounds_lgb,
                        feature_name = predictors,
                        categorical_feature = categorical)
    del dtrain_lgb
    gc.collect()

    print('--------------------------------------------------------------------') 
    print('>> save model...')
    # save model to file

    if not DEBUG:
        model_lgb.save_model(modelfilename+'.txt')

    # print('--------------------------------------------------------------------') 
    # print('>> reading test')
    # test_df = read_processed_h5(TEST_HDF5,predictors+['click_id'])
    # print(test_df.info()); print(test_df.head())
    # print_memory()
    # print("test size : ", len(test_df))
    # sub = pd.DataFrame()
    # sub['click_id'] = test_df['click_id'].astype('int')

    # print(">> predicting...")
    # sub['is_attributed'] = model_lgb.predict(test_df[predictors])
    # # if not debug:
    # print("writing...")
    # sub.to_csv(subfilename,index=False,compression='gzip')
    # print("done...")
    # return sub

# num_leaves_list = [16]
# max_depth_list = [-1]
# # option_list = [16, 15, 11, 10, 3]
# # option_list = [3, 15, 18]
# option_list = [15, 18]

# for option in option_list:
#     for i in range(len(num_leaves_list)):
#         print ('==============================================================')
#         num_leaves = num_leaves_list[i]
#         max_depth = max_depth_list[i]
#         print('num leaves:', num_leaves)
#         print('max depth:', max_depth)
#         if debug: print ('option:', option)
#         predictors = get_predictors(option)
#         subfilename = yearmonthdate_string + '_' + str(len(predictors)) + \
#                 'features_' + boosting_type + '_cv2_' + str(int(100*frac)) + \
#                 'percent_full_%d_%d'%(num_leaves,max_depth) + '_OPTION' + str(option) + '.csv.gz'
#         if debug: print (subfilename)                
#         if os.path.isfile(subfilename):
#             print('--------------------------------------')
#             print('Already trained...')
#         else:             
#             sub=DO(num_leaves,max_depth, option)

if __name__ == '__main__':
    main()