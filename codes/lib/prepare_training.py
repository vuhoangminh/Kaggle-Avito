import numpy as np
import h5py, time, gc
import pandas as pd
import pickle, datetime
from .read_write_file import load_pickle, read_train_test
from .print_info import print_doing, print_memory

def get_text_matrix(filename, dataset, debug, len_train): 
    with open(filename, "rb") as f:
        mat, tfvocab = pickle.load(f)   
    # mat, tfvocab  = load_pickle(filename)
    # mat = mat.todense()

    if debug: print(mat.shape, np.sum(mat))
    if dataset=='train':
        mat_ok = mat[0:len_train]
    elif dataset=='test':
        mat_ok = mat[len_train:]
    else:
        mat_ok = mat                
    print_memory()         
    return mat_ok, tfvocab                  

def read_processed_h5(filename, predictors, categorical):
    with h5py.File(filename,'r') as hf:
        feature_list = list(hf.keys())
    train_df = pd.DataFrame()
    t0 = time.time()
    for feature in feature_list:
        if feature in predictors:
            print('>> adding', feature)
            train_df[feature] = pd.read_hdf(filename, key=feature)
            print_memory()
    t1 = time.time()
    total = t1-t0
    print('total reading time:', total)
    return train_df    

def read_dataset(is_merged, debug):                   
    if debug:
        filename_train = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'train', debug)  
        filename_test = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'test', debug)                                                                    
    else:
        filename_train = '../input/{}.feather'.format('train')  
        filename_test = '../input/{}.feather'.format('test')  

    print_doing('reading train, test and merge')  
    if is_merged:  
        df = read_train_test(filename_train, filename_test, '.feather', is_merged=True)
        if debug: print(df.head())
    else:
        train_df, test_df = read_train_test(filename_train, filename_test, '.feather', is_merged=False)        
        if debug: print(train_df.head()); print(test_df.head())
    print_memory()
    if is_merged:
        return df
    else:
        return train_df, test_df 

def drop_col(df,cols):
    for col in cols:
        if col in df:
            df = df.drop([col], axis=1)
    return df

def load_train_test(index_col, target, debug):
    train_df, test_df = read_dataset(False, debug)
    len_train = len(train_df)

    train_df.set_index(index_col, inplace=True)
    traindex = train_df.index
    test_df.set_index(index_col, inplace=True)
    testdex = test_df.index
    
    train_labels = train_df.deal_probability.copy()
    train_df.drop(target, axis=1, inplace=True)

    # print(train_df.info(), train_df.head())
    # print(test_df.info(), test_df.head())  

    print('Train shape: {} Rows, {} Columns'.format(*train_df.shape))
    print('Test shape: {} Rows, {} Columns'.format(*test_df.shape))

    print("\n>> combine Train and Test")
    df = pd.concat([train_df,test_df],axis=0)
    # print(train_df.info())
    # print(test_df.info())
    # print(df.info())

    del train_df, test_df; gc.collect()
    print('\n data shape: {} Rows, {} Columns'.format(*df.shape))    

    return df, train_labels, len_train, traindex, testdex

def add_feature(df, filename):
    gp = load_pickle(filename)
    for feature in gp:
        df[feature] = gp[feature].values
    return df    

def get_string_time():
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
    return yearmonthdate_string

# def get_predictors(storename):
#     # with h5py.File(storename,'r') as hf:
#     #     feature_list = list(hf.keys())
#     # predictors = []
#     # for feature in feature_list:
#     #     if '_en' not in feature and feature not in REMOVED_LIST:
#     #         predictors = predictors + [feature]
#     return PREDICTORS    