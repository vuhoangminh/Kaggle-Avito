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
from lib.prepare_training import get_text_matrix, read_processed_h5, read_dataset, drop_col, load_train_test, add_feature, get_string_time
import lib.configs as configs
import features_list
from features_list import PREDICTORS

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

yearmonthdate_string = get_string_time()

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
    global args, DEBUG, FRAC, PREDICTORS, TRAINMODE
    args = parser.parse_args()
    DEBUG = args.debug
    FRAC = args.frac
    TRAINMODE = args.trainmode
    print_debug(DEBUG)
    DO()

def DO():
    tabular_predictors = get_tabular_predictors()
    if DEBUG:
        # mat_filename = '../processed_features_debug2/text_feature_kernel.pickle'
        mat_filename = '../processed_features_debug2/truncated_text_feature_kernel.pickle'
        dir_feature = '../processed_features_debug2/'
    else: 
        # mat_filename = '../processed_features/text_feature_kernel.pickle'
        mat_filename = '../processed_features/truncated_text_feature_kernel.pickle'
        dir_feature = '../processed_features/'
    X, y, test, full_predictors, predictors, testdex = prepare_training(mat_filename, dir_feature, tabular_predictors)
    categorical = get_categorical(predictors)
    predictors = get_predictors(predictors)

    # num_leave_list = [7,16,32]
    num_leave_list = [16]
    if TRAINMODE == 'gbdt':
        boosting_list = ['gbdt', 'dart']
    else:
        boosting_list = ['dart', 'gbdt']

    for boosting_type in boosting_list:
        for num_leave in num_leave_list:
            if DEBUG:
                subfilename = '../sub/debug_{}_{}_{}features_num_leave{}_OPTION{}.csv.gz'. \
                        format(yearmonthdate_string,boosting_type,str(len(predictors)),num_leave,1)
            else:
                subfilename = '../sub/{}_{}_{}features_num_leave{}_OPTION{}.csv.gz'. \
                        format(yearmonthdate_string,boosting_type,str(len(predictors)),num_leave,1)                        
            if os.path.exists(subfilename) and not DEBUG:
                print('{} done already'.format(subfilename))     
            else:                               
                model_lgb, subfilename = cv_train(X,y,num_leave,full_predictors,categorical,predictors,boosting_type,option=0)
                predict_sub(model_lgb, testdex, test, subfilename)
                del model_lgb; gc.collect()

def predict_sub(model_lgb, testdex, test, subfilename):
    print_doing('predicting')
    lgpred = model_lgb.predict(test)
    lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
    lgsub['deal_probability'].clip(0.0, 1.0, inplace=True)
    print('saving submission file to', subfilename)
    lgsub.to_csv(subfilename,index=True,header=True)
    print('done')

def cv_train(X,y,num_leave,full_predictors,categorical,predictors,boosting_type,option):

    if DEBUG: 
        subfilename = '../sub/debug_{}_{}_{}features_num_leave{}_OPTION{}.csv'. \
                format(yearmonthdate_string,boosting_type,str(len(predictors)),num_leave,option)
        modelfilename = '../trained_models/debug_{}_{}_{}features_num_leave{}_OPTION{}.txt'. \
                format(yearmonthdate_string,boosting_type,str(len(predictors)),num_leave,option)            
    else:           
        subfilename = '../sub/{}_{}_{}features_num_leave{}_OPTION{}.csv'. \
                format(yearmonthdate_string,boosting_type,str(len(predictors)),num_leave,option)
        modelfilename = '../trained_models/{}_{}_{}features_num_leave{}_OPTION{}.txt'. \
                format(yearmonthdate_string,boosting_type,str(len(predictors)),num_leave,option)

    print('\n----------------------------------------------------------')
    print("Training...")
    print('----------------------------------------------------------')

    start_time = time.time()

    params = {
        'boosting_type': boosting_type,
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.2,
        'num_leaves': num_leave,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'subsample': 0.8,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'feature_fraction': 0.8,  # Subsample ratio of columns when constructing each tree.
        'nthread': 4,
        'verbose': 0
    }

    print('>> prepare dataset...')
    dtrain_lgb = lgb.Dataset(X, y,
            feature_name=full_predictors,
            categorical_feature=categorical)                       
    print_memory()   

    print('params', params)
    print('\n>> start cv...')

    if DEBUG:
        num_boost_round = 300
        early_stopping_rounds = 10
    else:
        num_boost_round = 20000   
        early_stopping_rounds = 30                    
    cv_results  = lgb.cv(params, 
                        dtrain_lgb, 
                        categorical_feature = categorical,
                        num_boost_round=num_boost_round,                       
                        metrics='rmse',
                        seed = SEED,
                        shuffle = False,
                        nfold=10, 
                        show_stdv=True,
                        early_stopping_rounds=early_stopping_rounds, 
                        stratified=False,
                        verbose_eval=5)                     

    print('[{}]: model training time'.format(time.time() - start_time))
    print_memory()

    # print (cv_results)
    print('--------------------------------------------------------------------') 
    print("Model Report")
    num_boost_rounds_lgb = len(cv_results['rmse-mean'])
    print('num_boost_rounds_lgb = ' + str(num_boost_rounds_lgb))
    print('best rmse = {0:.4f}'.format(cv_results['rmse-mean'][num_boost_rounds_lgb-1]))
    
    print ('>> start trainning... ')
    model_lgb = lgb.train(
                        params, dtrain_lgb, 
                        num_boost_round=num_boost_rounds_lgb,
                        feature_name = full_predictors,
                        categorical_feature = categorical)
    del dtrain_lgb
    gc.collect()

    print('saving model to', modelfilename)
    model_lgb.save_model(modelfilename)
    
    return model_lgb, subfilename 

def prepare_training(mat_filename, dir_feature, predictors):
    df, train_labels, len_train, traindex, testdex = load_train_test(['item_id'], TARGET, DEBUG)
    df = drop_col(df,REMOVED_LIST)

    # add features
    print_doing('add tabular features')
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
    print_memory()

    # add text_feature
    print_doing('add text features')
    ready_df, tfvocab = get_text_matrix(mat_filename, 'all', 2, 0)

    # add text_feature
    print_doing('stack')   
    X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
    testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])

    tfvocab = df.columns.tolist() + tfvocab
    for shape in [X,testing]:
        print("{} Rows and {} Cols".format(*shape.shape))
    print("Feature Names Length: ",len(tfvocab))
    # del df; gc.collect()  

    return X, train_labels, testing, tfvocab, df.columns.tolist(), testdex  

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