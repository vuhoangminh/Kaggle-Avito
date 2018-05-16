import pickle, os
import pandas as pd
import numpy as np
import random

from lib.read_write_file import save_csv, save_feather, save_file, save_pickle
from lib.read_write_file import load_csv, load_feather, load_pickle, read_train_test

from sklearn.model_selection import train_test_split
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
        


# storename = featuredir + 'test_debug2.h5'

# store = pd.HDFStore(storename) 
# existing_key = store.keys()
# store.close()
# df = pd.DataFrame()
# for feature in existing_key:
#     df[feature] = pd.read_hdf(storename, key=feature)  
#     print('reading', feature)

# print(df.isnull().sum(axis=0))  


# df = pd.DataFrame({"fruit":["apple","banana","apple","ban"],
#         "weight":[7,8,3,8],"price":[4,5,6,6]})   
# print(df)       

# df['size'] = df.groupby(['fruit','price']).transform(np.size)       
# print(df)

# df['freq'] = df.groupby('fruit')['fruit'].transform('count')
# print(df)

# selcols = ['fruit']

# df5 = df[selcols].groupby(selcols).size().reset_index(name="Time4")
# print(df5)
# df = df.merge(df5, on=selcols, how='left')
# print(df)

# feature_name = 'Freq'
# df6 = df[selcols]. \
#     groupby(selcols).size(). \
#     reset_index(name=feature_name)

# print(df6)
# df = df.merge(df6, on=selcols, how='left')
# print(df)


filename = '../input/aggregated_features.csv'
df = load_csv(filename)

savename = '../input/kernel_aggregated_features.pickle'
save_pickle(df, savename)