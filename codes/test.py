import pickle, os
import pandas as pd

from lib.read_write_file import save_csv, save_feather, save_file, save_pickle
from lib.read_write_file import load_csv, load_feather, load_pickle, read_train_test

import glob

cwd = os.getcwd()
print ('working dir', cwd)
if 'codes' not in cwd:
    default_path = cwd + '/codes/'
    os.chdir(default_path)

DEBUG = 2

if DEBUG:
    featuredir = '../processed_features_debug{}/'.format(DEBUG)
else:
    featuredir = '../processed_features/'

files = glob.glob(featuredir + '*.pickle')



# for file in files:
#     if 'text_feature_kernel' not in file:
#         print(file)
#         filename = file
#         print('\n>> doing', filename)
#         df = load_pickle(filename)
#         print(df.isnull().sum(axis=0))
        


storename = featuredir + 'test_debug2.h5'

store = pd.HDFStore(storename) 
existing_key = store.keys()
store.close()
df = pd.DataFrame()
for feature in existing_key:
    df[feature] = pd.read_hdf(storename, key=feature)  
    print('reading', feature)

print(df.isnull().sum(axis=0))                     
