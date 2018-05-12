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
        if key in existing_key:
            print ('feature already added...')
        else:                    
            print ('add {} to {}'.format(feature, storename))  
            temp = pd.DataFrame()
            if feature in DATATYPE_LIST:
                temp[feature] = df[feature].astype(DATATYPE_LIST[feature])
            else:
                temp[feature] = df[feature]  
                                     

            # if not os.path.exists(storename):
            #     f = h5py.File(storename, 'w')
            # # else:                
            # with h5py.File(storename) as hf:
            #     hf[feature] = temp         
            #     # temp.to_hdf(storename, key=feature, mode='a')

            # with h5py.File(storename) as f:
            #     f[feature] = temp

            store = pd.HDFStore(storename) 
            store[feature] = temp
            store.close()


            get_info_key_hdf5(storename, key=feature)  



# with h5py.File("some_path.h5") as f:
#    f["data1"] = some_data



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
           