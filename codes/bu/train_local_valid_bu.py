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
from features_list import PREDICTORS_BASED, PREDICTORS_OVERFIT, PREDICTORS_GOOD, PREDICTORS_NOTCHECKED

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
    global args, DEBUG, FRAC, PREDICTORS, TRAINMODE, PREDICTORS, LOCAL_VALIDATION_RESULT
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

    option_list = []
    for option in range(8):
        option_list = option_list + ['option' + str(option)]

    LOCAL_VALIDATION_RESULT = pd.DataFrame(index=option_list, 
            columns=['running_time','num_round','train','val','local_test'])

    if DEBUG: print(option_list); print(LOCAL_VALIDATION_RESULT)

    for option in range(8):
        # nothing here
        if option == 0:
            is_textadded = False
            PREDICTORS = PREDICTORS_BASED
            mat_filename = dir_feature + 'text_feature_kernel.pickle'
        # kernel    
        elif option == 1:
            is_textadded = True
            PREDICTORS = PREDICTORS_BASED
            mat_filename = dir_feature + 'text_feature_kernel.pickle'
        # kernel max_feature = 1000               
        elif option == 2:
            is_textadded = True
            PREDICTORS = PREDICTORS_BASED
            mat_filename = dir_feature + 'text_feature_kernel_1000.pickle'
        # kernel max_feature = 30000            
        elif option == 3:   
            is_textadded = True
            PREDICTORS = PREDICTORS_BASED
            mat_filename = dir_feature + 'text_feature_kernel_30000.pickle'                 
        # kernel max_feature = infinite            
        elif option == 4:   
            is_textadded = True
            PREDICTORS = PREDICTORS_BASED
            mat_filename = dir_feature + 'text_feature_kernel_-1.pickle'                 
        # kernel max_feature = 18000 + 'good' feature
        elif option == 5:
            is_textadded = True
            PREDICTORS = PREDICTORS_BASED + PREDICTORS_GOOD
            mat_filename = dir_feature + 'text_feature_kernel.pickle'
        # kernel max_feature = 18000 + not-checked feature            
        elif option == 6:
            is_textadded = True
            PREDICTORS = PREDICTORS_BASED + PREDICTORS_NOTCHECKED
            mat_filename = dir_feature + 'text_feature_kernel.pickle'
        elif option == 7:
            is_textadded = True
            PREDICTORS = PREDICTORS_BASED + PREDICTORS_OVERFIT
            mat_filename = dir_feature + 'text_feature_kernel.pickle'
        if DEBUG:
            print_header('Option {}'.format(option))
            print('is_textadded {} \n predictors {} \n mat filename {}'.format(is_textadded, PREDICTORS, mat_filename))

        DO(option, is_textadded, mat_filename, dir_feature)  


    print_header('FINAL SUMMARY')
    print(LOCAL_VALIDATION_RESULT)

def DO(option, is_textadded, mat_filename, dir_feature):
    tabular_predictors = get_tabular_predictors()
       
    X, y, test, full_predictors, predictors, testdex = prepare_training(mat_filename, dir_feature, 
            tabular_predictors, is_textadded=is_textadded)
    categorical = get_categorical(predictors)
    predictors = get_predictors(predictors)

    if TRAINMODE == 'gbdt':
        boosting_list = ['gbdt']
    else:
        boosting_list = ['dart']

    for boosting_type in boosting_list:
        if DEBUG:
            subfilename = '../sub/debug_{}_{}_{}features_num_leave{}_OPTION{}.csv.gz'. \
                    format(yearmonthdate_string,boosting_type,str(len(predictors)),-1,option)
        else:
            subfilename = '../sub/{}_{}_{}features_num_leave{}_OPTION{}.csv.gz'. \
                    format(yearmonthdate_string,boosting_type,str(len(predictors)),-1,option)                        
        if os.path.exists(subfilename) and not DEBUG:
            print('{} done already'.format(subfilename))     
        else:                               
            model_lgb, subfilename = train(X,y,-1,full_predictors,
                    categorical,predictors,boosting_type,option=option)
            predict_sub(model_lgb, testdex, test, subfilename)
            del model_lgb; gc.collect()

def predict_sub(model_lgb, testdex, test, subfilename):
    print_header('Submission')
    print_doing_in_task('predicting')
    lgpred = model_lgb.predict(test)
    lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
    lgsub['deal_probability'].clip(0.0, 1.0, inplace=True)
    print('saving submission file to', subfilename)
    lgsub.to_csv(subfilename,index=True,header=True)
    print('done')

def train(X,y,num_leave,full_predictors,categorical,predictors,boosting_type,option):
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

    print_header("Training")
    start_time = time.time()               

    print_doing_in_task('prepare dataset...')

    X, X_local_valid, y, y_local_valid = train_test_split(
        X, y, test_size=0.10, random_state=SEED)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=SEED)

    print('training shape: {} \n'.format(X.shape))

    print("Light Gradient Boosting Regressor")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': boosting_type,
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 15,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'learning_rate': 0.1,
        'verbose': 0
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
        early_stopping_rounds = 30  

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
    val_rmse = '{0:.4f}'.format(np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
    train_rmse = '{0:.4f}'.format(np.sqrt(metrics.mean_squared_error(y_train, lgb_clf.predict(X_train))))
    local_valid_rmse = '{0:.4f}'.format(np.sqrt(metrics.mean_squared_error(y_local_valid, lgb_clf.predict(X_local_valid))))

    print('OPTION', option)
    print('model training time:     {} mins'.format(runnning_time))
    print('num_boost_rounds_lgb:    {}'.format(num_boost_rounds_lgb))
    print('train rmse:              {}'.format(train_rmse))
    print('val rmse:                {}'.format(val_rmse))
    print('local valid rmse:        {}'.format(local_valid_rmse))

    print('saving model to', modelfilename)
    lgb_clf.save_model(modelfilename)

    option_name = 'option' + str(option)
    LOCAL_VALIDATION_RESULT['running_time'][option_name] = runnning_time   
    LOCAL_VALIDATION_RESULT['num_round'][option_name] = num_boost_rounds_lgb   
    LOCAL_VALIDATION_RESULT['train'][option_name] = train_rmse  
    LOCAL_VALIDATION_RESULT['val'][option_name] = val_rmse  
    LOCAL_VALIDATION_RESULT['local_test'][option_name] = local_valid_rmse
    return lgb_clf, subfilename 

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

def create_local_validation():
    if DEBUG:
        dir_feature = '../processed_features_debug2/'
        N = 1503
    else: 
        dir_feature = '../processed_features/' 
        N = 1503424 
    filename = dir_feature + 'local_validation.pickle'
    if os.path.exists(filename):
        local_validation_array = load_pickle(filename)
    else:  
        local_validation_array = [i for i in range(0,N,int(N/100))]
        save_pickle(local_validation_array, filename)
    return local_validation_array       

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