import mlcrate as mlc
import pandas as pd
import pickle, os
from .print_info import print_doing
   

def save_pickle(df, filename):
    print_doing('saving to {}'.format(filename))
    with open(filename, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL) 

def save_feather(df, filename):
    print_doing('saving to {}'.format(filename))
    mlc.save(df, filename)  

def save_csv(df, filename, is_sub=False):
    print_doing('saving to {}'.format(filename))
    if is_sub:
        df.to_csv(filename,index=False,compression='gzip')
    else:
        df.to_csv(filename,index=False)                

def load_pickle(filename):
    print_doing('read {}'.format(filename))
    df = pickle.load(open(filename, "rb" ))
    return df

def load_feather(filename):
    print_doing('read {}'.format(filename))
    df = mlc.load(filename)
    return df

def load_csv(filename, nrows=0):
    print_doing('read {}'.format(filename))
    if nrows>0:
        df = pd.read_csv(filename, 
                nrows=nrows)
    else:
        df = pd.read_csv(filename)
    return df        

def read_train_test(filename_train, filename_test, ext, is_merged):
    if ext=='.csv':
        train_df = load_csv(filename_train)
        test_df = load_csv(filename_test)
    elif ext=='.feather':
        train_df = load_feather(filename_train)
        test_df = load_feather(filename_test)  
    elif ext=='.pickle':
        train_df = load_pickle(filename_train)
        test_df = load_pickle(filename_test)  
    if is_merged:
        train_df = train_df.append(test_df)
        train_df = train_df.fillna(0)
        return train_df 
    else:
        return train_df, test_df

def save_file(df, filename, ext):
    if ext == '.pickle':
        save_pickle(df, filename)
    elif ext == '.feather':
        save_feather(df, filename)
    else:
        save_csv(df, filename)

def load_file(filename, ext):
    if ext == '.pickle':
        df = load_pickle(filename)
    elif ext == '.feather':
        df = load_feather(filename)
    else:
        df = load_csv(filename)
    return df                