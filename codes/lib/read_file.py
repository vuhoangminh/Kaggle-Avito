import mlcrate as mlc
import panda as pd
import pickle

def load_pickle(filename):
    df = pickle.load(open(filename, "rb" ))
    return df

def load_feather(filename):
    return mlc.load(filename)

def load_csv(filename, nrows=0):
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
                        