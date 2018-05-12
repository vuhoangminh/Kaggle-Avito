import pandas as pd
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
    for feature in usecols:
        key = '/' + feature
        print('>> doing', key)
        if key in existing_key:
            print ('feature already added...')
        else:                    
            print ('add key to hdf5...')                        
            temp = pd.DataFrame()
            if feature in DATATYPE_LIST:
                temp[feature] = df[feature].astype(DATATYPE_LIST[feature])
            else:
                temp[feature] = df[feature]                                
            temp.to_hdf(storename, key=feature, mode='a')
            get_info_key_hdf5(storename, key=feature)  


           