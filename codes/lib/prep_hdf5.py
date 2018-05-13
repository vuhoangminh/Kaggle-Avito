import pandas as pd
import h5py
import os
from .configs import DATATYPE_DICT, DATATYPE_LIST

def get_datatype(feature):
    datatype = 'UNKNOWN'
    for key, type in DATATYPE_DICT.items():
        if key in feature:
            datatype = type
            break
    return datatype 

def get_info_key_hdf5(storename, key):
    print('-----------------------')
    store = pd.HDFStore(storename)
    print(store)
    df = store.select(key) 
    print('-----------------------')
    print(df.info())

def add_dataset_to_hdf5(storename, df):
    usecols = list(df)
    store = pd.HDFStore(storename) 
    existing_key = store.keys()
    store.close()
    for feature in usecols:
        key = '/' + feature
        print('\n>> doing', key)
        temp = pd.DataFrame()
        if key in existing_key:
            print ('feature already added...')
            temp[feature] = pd.read_hdf(storename, key=feature)                
        else:                    
            print ('add {} to {}'.format(feature, storename))  
            temp[feature] = df[feature]  
            store = pd.HDFStore(storename) 
            store[feature] = temp
            store.close()
            get_info_key_hdf5(storename, key=feature)  
    return temp            



def add_text_feature_to_hdf5(storename, mat, language):
    store = pd.HDFStore(storename) 
    existing_key = store.keys()
    if language == 'russian':
        key = '/' + 'dense'
        feature = 'dense'
    else:
        key = '/' + 'dense_en'                
        feature = 'dense_en'
    print('\n>> doing', key)
    if key in existing_key:
        print ('feature already added...')
    else:                    
        print ('add {} to {}'.format(feature, storename))  

        h5f = h5py.File(storename, 'a')
        h5f.create_dataset(feature, data=mat)
        get_info_key_hdf5(storename, key=feature)     
           