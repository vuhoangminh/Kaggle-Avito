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
from features_list import PREDICTORS_BASED, PREDICTORS_OVERFIT, PREDICTORS_GOOD, PREDICTORS_NOTCHECKED, PREDICTORS_TRY

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
# parser.add_argument('-o', '--option', default=0, type=int) 

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
    global args, DEBUG, FRAC, PREDICTORS, TRAINMODE, PREDICTORS, LOCAL_TUNE_RESULT
    args = parser.parse_args()
    DEBUG = args.debug
    FRAC = args.frac
    TRAINMODE = args.trainmode
    # OPTION=args.option
    print_debug(DEBUG)

    if DEBUG:
        dir_feature = '../processed_features_debug2/' 
    else:
        dir_feature = '../processed_features/'            

    # boosting_list = ['gbdt', 'dart']
    boosting_list = ['gbdt']
    num_leave_list = [7,9,15,31,63,128]
    max_depth_list = [3,4,7,15,31,64]

    model_list = []
    for i in range(len(num_leave_list)):
        num_leave = num_leave_list[i]
        max_depth = max_depth_list[i]
        for boosting_type in boosting_list:
            model_list = model_list + ['{}_{}_{}'.format(boosting_type,num_leave,max_depth)]


    LOCAL_TUNE_RESULT = pd.DataFrame(index=model_list,
            columns=['running_time','num_round','train','val'])
    if DEBUG: print(LOCAL_TUNE_RESULT)

    option = 1
    is_textadded = True
    PREDICTORS = PREDICTORS_BASED
    mat_filename = dir_feature + 'text_feature_kernel.pickle'                 
    print_header('Option {}'.format(option))
    print('is_textadded {} \n predictors {} \n mat filename {}'.format(is_textadded, PREDICTORS, mat_filename))

    for k in range(len(num_leave_list)):
        i = len(num_leave_list) - k - 1
        num_leave = num_leave_list[i]
        max_depth = max_depth_list[i]
        for boosting_type in boosting_list:
            DO(option, is_textadded, mat_filename, dir_feature, num_leave, max_depth, boosting_type) 

    print_header('FINAL SUMMARY')
    print(LOCAL_TUNE_RESULT)
    LOCAL_TUNE_RESULT.to_csv('csv/tune_params.csv', index=True)

def DO(option, is_textadded, mat_filename, dir_feature, num_leave, max_depth, boosting_type):
    tabular_predictors = get_tabular_predictors(PREDICTORS)
       
    X, y, test, full_predictors, predictors, testdex = prepare_training(mat_filename, dir_feature, 
            tabular_predictors, is_textadded=is_textadded)
    categorical = get_categorical(predictors)
    predictors = get_predictors(predictors)

    train(X,y,num_leave,max_depth,full_predictors,
                    categorical,predictors,boosting_type,option=option)
    gc.collect()

def train(X,y,num_leave,max_depth,full_predictors,categorical,predictors,boosting_type,option):

    print_header("Training")
    start_time = time.time()               

    print_doing_in_task('prepare dataset...')
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=SEED)

    print('training shape: {} \n'.format(X.shape))

    print("Light Gradient Boosting Regressor")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': boosting_type,
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': max_depth,
        'num_leave': num_leave,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'learning_rate': 0.1,
        'lambda_l1' : 10,
        'max_bin' : 512,
        'verbose': -1
    }  
    print('params:', lgbm_params)

    lgtrain = lgb.Dataset(X_train, y_train,
                    feature_name=full_predictors,
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                    feature_name=full_predictors,
                    categorical_feature = categorical)

    if DEBUG:
        num_boost_round = 300
        early_stopping_rounds = 10
    else:
        num_boost_round = 20000   
        early_stopping_rounds = 100

    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=num_boost_round,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10
    )

    print_memory()

    print_header("Model Report")

    runnning_time = '{0:.2f}'.format((time.time() - start_time)/60)
    num_boost_rounds_lgb = lgb_clf.best_iteration
    print_doing_in_task('fit val')
    val_rmse = '{0:.4f}'.format(np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
    print_doing_in_task('fit train')
    train_rmse = '{0:.4f}'.format(np.sqrt(metrics.mean_squared_error(y_train, lgb_clf.predict(X_train))))    
    print_header("Model Report")
    
    print('boosting_type {}, num_leave {}, max_depth {}'.format(boosting_type,num_leave,max_depth))
    print('model training time:     {0:.2f} mins'.format((time.time() - start_time)/60))
    print('num_boost_rounds_lgb:    {}'.format(lgb_clf.best_iteration))
    print('best rmse:               {0:.4f}'.format(np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid)))))

    model = '{}_{}_{}'.format(boosting_type,num_leave,max_depth)
    LOCAL_TUNE_RESULT['running_time'][model] = runnning_time   
    LOCAL_TUNE_RESULT['num_round'][model] = num_boost_rounds_lgb   
    LOCAL_TUNE_RESULT['train'][model] = train_rmse  
    LOCAL_TUNE_RESULT['val'][model] = val_rmse  

def prepare_training(mat_filename, dir_feature, predictors, is_textadded):
    print_header('Load features')
    df, y, len_train, traindex, testdex = load_train_test(['item_id'], TARGET, DEBUG)
    del len_train; gc.collect()
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
                print_doing_in_task('adding {}'.format(feature))
                df = add_feature(df, dir_feature_file)
    print_memory()

    if is_textadded:
        # add text_feature
        print_doing_in_task('add text features')
        ready_df, tfvocab = get_text_matrix(mat_filename, 'all', 2, 0)

        # stack
        print_doing_in_task('stack')   
        X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
        testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])
        print_memory()

        print_doing_in_task('prepare vocab')   
        tfvocab = df.columns.tolist() + tfvocab
        for shape in [X,testing]:
            print("{} Rows and {} Cols".format(*shape.shape))
        print("Feature Names Length: ",len(tfvocab))
    
    else:
        tfvocab = df.columns.tolist()
        testing = hstack([csr_matrix(df.loc[testdex,:].values)])
        X = hstack([csr_matrix(df.loc[traindex,:].values)]) # Sparse Matrix

    return X, y, testing, tfvocab, df.columns.tolist(), testdex  

def get_tabular_predictors(predictors):
    predictors = predictors
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