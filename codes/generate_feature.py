import argparse
import pandas as pd
import sys, os
import textblob
from tqdm import tqdm,tqdm_pandas
import math
import mlcrate as mlc
import gc
import numpy as np
from googletrans import Translator
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import time
import nltk, textwrap
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from sklearn.model_selection import train_test_split

from lib.print_info import print_debug, print_doing, print_memory, print_doing_in_task
from lib.read_write_file import save_csv, save_feather, save_file, save_pickle
from lib.read_write_file import load_csv, load_feather, load_pickle, read_train_test
from lib.gen_feature import generate_groupby_by_type_and_columns, create_time, measure_length, map_key, create_text_feature, create_label_encode
import lib.configs as configs
import features_list

SEED = configs.SEED
DATATYPE_DICT = configs.DATATYPE_DICT
NAMEMAP_DICT = configs.NAMEMAP_DICT
MINH_LIST_MEAN_DEAL_PROB = features_list.MINH_LIST_MEAN_DEAL_PROB
MINH_LIST_MEAN_PRICE = features_list.MINH_LIST_MEAN_PRICE
MINH_LIST_VAR_DEAL_PROB = features_list.MINH_LIST_VAR_DEAL_PROB
MINH_LIST_VAR_PRICE = features_list.MINH_LIST_VAR_PRICE

cwd = os.getcwd()
print ('working dir', cwd)
if 'codes' not in cwd:
    default_path = cwd + '/codes/'
    os.chdir(default_path)

parser = argparse.ArgumentParser(
    description='translate',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--debug', default=2, type=int, choices=[0,1,2])    

parser.add_argument('-d', '--dataset', type=str, default='train_translated',
    choices=['train_translated','test_translated','train_active_translated',
    'test_active_translated','train','test'])
# parser.add_argument('-d', '--dataset', type=str, default='train',choices=['train','test'])

def main():
    global args, DEBUG, DATASET
    args = parser.parse_args()
    DATASET = args.dataset
    DEBUG = args.debug
    print_debug(DEBUG)
    df = read_dataset(DATASET)
    df_good = read_dataset_deal_probability(DATASET, 1988)
    if DEBUG:
        save_csv(df, 'df.csv')
        save_csv(df_good, 'df_good.csv')
    if DEBUG:
        todir = '../processed_features_debug{}/'.format(DEBUG)
    else:
        todir = '../processed_features/'

    ## done
    # gen_label_encode(df, todir, '.pickle')
    # gen_time_feature(df, todir, '.pickle')
    gen_mean_deal_probability (df_good, todir, '.pickle')
    # gen_mean_price (df, todir, '.pickle')
    # gen_var_deal_probability (df_good, todir, '.pickle')
    # gen_var_price (df, todir, '.pickle')
    # gen_text_feature_from_kernel (df, todir, '.pickle', 'russian', -1)
    # gen_text_feature_from_kernel (df, todir, '.pickle', 'russian', 1000)
    # gen_text_feature_from_kernel (df, todir, '.pickle', 'russian', 30000)
    gen_text_feature_from_kernel (df, todir, '.pickle', 'russian', 18000)
    
    ## after translated!!
    # gen_text_feature_from_kernel (df, todir, '.pickle', 'english')
    # gen_len_title_description_feature(df, todir, '.pickle') 
    # get_svdtruncated_vectorizer(todir)
    
def gen_label_encode(df, todir, ext):
    gp = create_label_encode(df, todir, ext)
    if DEBUG: print(df.head()), print (gp.head())
    del gp; gc.collect()
    print_memory()

def gen_text_feature_from_kernel(df, todir, ext, language, max_features):
    create_text_feature (df, todir, ext, language, max_features)
    print_memory()    
    
def gen_time_feature(df, todir, ext):
    gp = create_time(df, todir=todir, ext = ext)
    if DEBUG: print(df['activation_date'].head()), print (gp.head())
    del gp; gc.collect()
    print_memory()

def gen_len_title_description_feature(df, todir, ext):
    selcols = ['title_en','description_en','title','description']
    gp = measure_length(df, selcols=selcols, todir=todir, ext = '.pickle')
    if DEBUG: print(df[selcols].head()), print (gp.head())
    del gp; gc.collect()    
    print_memory()

def gen_mean_deal_probability (df, todir, ext):
    for selcols in MINH_LIST_MEAN_DEAL_PROB:
        gp = generate_groupby_by_type_and_columns(df, selcols, 'mean', todir, ext)
        if DEBUG: print(df[selcols].head()), print (gp.head())
        del gp; gc.collect()    
        print_memory()

def gen_mean_price (df, todir, ext):
    for selcols in MINH_LIST_MEAN_PRICE:
        gp = generate_groupby_by_type_and_columns(df, selcols, 'mean', todir, ext)
        if DEBUG: print(df[selcols].head()), print (gp.head())
        del gp; gc.collect()    
        print_memory()

def gen_var_deal_probability (df, todir, ext):
    for selcols in MINH_LIST_VAR_DEAL_PROB:
        gp = generate_groupby_by_type_and_columns(df, selcols, 'var', todir, ext)
        if DEBUG: print(df[selcols].head()), print (gp.head())
        del gp; gc.collect()    
        print_memory()

def gen_var_price (df, todir, ext):
    for selcols in MINH_LIST_VAR_PRICE:
        gp = generate_groupby_by_type_and_columns(df, selcols, 'var', todir, ext)
        if DEBUG: print(df[selcols].head()), print (gp.head())
        del gp; gc.collect()    
        print_memory()

def read_dataset(dataset):                   
    debug = DEBUG
    if debug:
        filename_train = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'train', debug)  
        filename_test = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'test', debug)                                                                    
    else:
        filename_train = '../input/{}.feather'.format('train')  
        filename_test = '../input/{}.feather'.format('test')  

    print_doing('reading train, test and merge')    
    df = read_train_test(filename_train, filename_test, '.feather', is_merged=1)
    print_memory()
    print(df.head())
    return df

def read_dataset_deal_probability(dataset, seed):                   
    debug = DEBUG
    if debug:
        filename_train = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'train', debug)  
        filename_test = '../input/debug{}/{}_debug{}.feather'.format(
                debug, 'test', debug)                                                                    
    else:
        filename_train = '../input/{}.feather'.format('train')  
        filename_test = '../input/{}.feather'.format('test')  

    print_doing('reading train, test and merge')    
    train_df, test_df = read_train_test(filename_train, filename_test, '.feather', is_merged=0)
    df = find_df_local_valid_and_make_deal_prob_nan(train_df, test_df, seed)
    print_memory()
    print(df.head())
    return df

def get_svdtruncated_vectorizer(todir):
    print_doing('doing svdtruncated text feature')
    filename = todir + 'text_feature_kernel.pickle'
    savename = todir + 'truncated_text_feature_kernel.pickle'
    if os.path.exists(savename):
        print('done already...')
        with open(savename, "rb") as f:
            svd_matrix,vocab = pickle.load(f)
        with open(filename, "rb") as f:
            tfid_matrix, tfvocab = pickle.load(f)             
    else:        
        with open(filename, "rb") as f:
            tfid_matrix, tfvocab = pickle.load(f)  
        svdT = TruncatedSVD(n_components=400)
        print_doing_in_task('truncated svd')
        svd_matrix = svdT.fit_transform(tfid_matrix)
        print_doing_in_task('convert to sparse')
        svd_matrix = sparse.csr_matrix(svd_matrix, dtype=np.float32)
        vocab = []
        for i in range(np.shape(svd_matrix)[1]):
            vocab.append('lsa'+str(i+1))
        with open(savename, "wb") as f:
            pickle.dump((svd_matrix,vocab), f, protocol=pickle.HIGHEST_PROTOCOL) 
    print('---- before truncate')
    print(tfid_matrix.shape), print ('len of feature:', len(tfvocab))
    print('---- after truncate')
    print(svd_matrix.shape), print ('len of feature:', len(vocab))

    if DEBUG:
        print(tfid_matrix)
        print('\n')
        print(svd_matrix)
    
    del svd_matrix, vocab, tfid_matrix, tfvocab; gc.collect()    
    print_memory()            
         
def read_dataset_origin(dataset):                   
    filename_train = '../input/train.csv'
    filename_test = '../input/test.csv'
    print_doing('reading train, test and merge')    
    df = read_train_test(filename_train, filename_test, '.feather', is_merged=1)
    print_memory()
    print(df.head())
    return df

def find_df_local_valid_and_make_deal_prob_nan(train_df, test_df, seed):
    train_df_train, train_df_valid = train_test_split(train_df, test_size=0.1, random_state=seed)
    print(train_df); print(train_df.info())
    print(train_df_train); print(train_df_train.info())
    print(train_df_valid); print(train_df_valid.info())
    train_df_valid['deal_probability'] = np.nan
    train_df_good = pd.concat([train_df_train, train_df_valid], axis=0)
    print(train_df_valid); print(train_df_valid.info())
    train_df_good.sort_index(inplace=True)
    print(train_df_good); print(train_df_good.info())
    df = pd.concat([train_df_good, test_df], axis=0)

    return df

def test():
    print (map_key('user_id'))
    print (map_key('abc'))

if __name__ == '__main__':
    main()
    # test()