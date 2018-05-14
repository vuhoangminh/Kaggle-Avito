import os, sys, inspect
import pandas as pd
import gc, pickle
import numpy as np
from .read_write_file import save_file, load_file
from .configs import NAMEMAP_DICT, TRANSLATE_LIST
from .print_info import print_memory

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
import nltk
from sklearn import preprocessing
nltk.download('stopwords')

# NAMEMAP_DICT = configs.NAMEMAP_DICT 

def map_key(key):
    if key in NAMEMAP_DICT:
        return NAMEMAP_DICT[key]
    else:
        return key

def generate_groupby_by_type_and_columns(df, selcols, apply_type, todir, ext):      
    feature_name = ''

    if apply_type != 'freq': 
        for i in range(len(selcols)-1):
            feature_name = feature_name + map_key(selcols[i]) + '_'
        feature_name = feature_name + apply_type + '_' + map_key(selcols[len(selcols)-1])
    else:
        for i in range(len(selcols)):
            feature_name = feature_name + map_key(selcols[i]) + '_'
        feature_name = feature_name + apply_type

    print('\n>> doing feature:', feature_name)
    
    filename = todir + feature_name + ext

    if os.path.exists(filename):
        print ('done already...')
        col_extracted = load_file(filename, ext)
    else:
        if apply_type == 'freq':
            col_temp = df[selcols]. \
                groupby(selcols).size(). \
                reset_index(name=feature_name)
        if apply_type == 'count':
            col_temp = df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count(). \
                reset_index().rename(index=str, columns={selcols[len(selcols)-1]: feature_name})
        if apply_type == 'var':
            col_temp = df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var(). \
                reset_index().rename(index=str, columns={selcols[len(selcols)-1]: feature_name})
        if apply_type == 'std':
            col_temp = df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].std(). \
                reset_index().rename(index=str, columns={selcols[len(selcols)-1]: feature_name})   
        if apply_type == 'cumcount':
            col_temp = df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount()
        if apply_type == 'nunique':
            col_temp = df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique(). \
                reset_index().rename(index=str, columns={selcols[len(selcols)-1]: feature_name})   
        if apply_type == 'mean':
            col_temp = df[selcols]. \
                groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean(). \
                reset_index().rename(index=str, columns={selcols[len(selcols)-1]: feature_name})   

        col_extracted = pd.DataFrame()
        if apply_type == 'cumcount':
            col_extracted[feature_name] = col_temp.values
            del col_temp; gc.collect()            
        elif apply_type == 'freq':
            df = df.merge(col_temp, on=selcols, how='left')
            del col_temp; gc.collect()
            col_extracted[feature_name] = df[feature_name]            
        else:
            df = df.merge(col_temp, on=selcols[0:len(selcols)-1], how='left')
            del col_temp; gc.collect()
            col_extracted[feature_name] = df[feature_name]

        save_file(df=col_extracted,filename=filename,ext=ext)                      
    return col_extracted 

def create_time(df, todir, ext):
    print('\n>> extract time')
    filename = todir + 'time' + ext
    if os.path.exists(filename):
        print ('done already...')
        gp = load_file(filename, ext)
    else:
        gp = pd.DataFrame()
        gp['week'] = pd.to_datetime(df.activation_date).dt.week.astype('uint8')
        gp['weekday'] = pd.to_datetime(df.activation_date).dt.weekday.astype('uint8')
        gp['day'] = pd.to_datetime(df.activation_date).dt.day.astype('uint8')
        # print('\n >> saving to', filename)
        save_file(df=gp, filename=filename, ext=ext)  
    return gp              

def measure_length(df, selcols, todir, ext):
    print('\n>> extract len of', selcols)
    filename = todir + 'len_title_description' + ext
    if os.path.exists(filename):
        print ('done already...')
        gp = load_file(filename, ext)
    else:    
        gp = pd.DataFrame()
        for col in selcols:
            new_feature = 'len_' + col
            gp[new_feature] = df[col].str.len()
        gp.fillna(1, inplace=True)    
        gp = gp.astype('int')
        # print('\n >> saving to', filename)
        save_file(df=gp, filename=filename, ext=ext)  
    return gp.astype('int')

def get_col(col): 
    return lambda x: x[col]

def get_original_data(df, selcols):
    for col in selcols:
        df[col] = df[col].replace({'0':np.nan, 0:np.nan})
    return df        

def remove_english(df):
    feature_list = list(df.keys())
    for feature in feature_list:
        if feature not in list(TRANSLATE_LIST.keys()):
            df = df.drop([feature], axis=1)
    return df

def remove_russian(df):
    feature_list = list(df.keys())
    for feature in feature_list:
        if feature not in list(TRANSLATE_LIST.values()):
            df = df.drop([feature], axis=1)
    return df           

def rename_en_to_ru(df):
    TRANSLATE_LIST_SWITCHED = {y:x for x,y in TRANSLATE_LIST.items()}
    df.columns = df.columns.to_series().map(TRANSLATE_LIST_SWITCHED)  
    return df  

def create_text_feature (df, todir, ext, language, max_features):
    print('\n>> doing Text Features')
    if language == 'russian':
        df = remove_english(df)
    else:
        df = remove_russian(df)
        df = rename_en_to_ru(df)

    if language == 'russian':
        filename = todir + 'text_feature_kernel_' + str(max_features) + ext
        filename_len = todir + 'len_feature_kernel' + ext  
        filename_vocab = todir + 'vocab_' + str(max_features) + ext  
        suffix_feature = ''
    else:
        filename = todir + 'text_feature_kernel_' + str(max_features) + '_en' + ext   
        filename_len = todir + 'len_feature_kernel_en' + ext     
        filename_vocab = todir + 'vocab_' + str(max_features) + '_en' + ext  
        suffix_feature = '_en'

    if os.path.exists(filename):
        print ('done already...')
        df = load_file(filename_len, ext)
        ready_df = load_file(filename, ext)
    else:
        df = get_original_data(df, ['param_1', 'param_2', 
                'param_3', 'description', 'title'])              

        df['text_feat'] = df.apply(lambda row: ' '.join([
            str(row['param_1']), 
            str(row['param_2']), 
            str(row['param_3'])]),axis=1) # Group Param Features

        print (df[['text_feat', 'param_1', 'param_2', 'param_3']].head())
        print (df[['text_feat', 'param_1', 'param_2', 'param_3']].tail())
        df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)        
        textfeats = ["description","text_feat", "title"]
        for cols in textfeats:
            df[cols] = df[cols].astype(str) 
            df[cols] = df[cols].astype(str).fillna('n/a') # FILL NA
            df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
            df[cols + '_num_chars' + suffix_feature] = df[cols].apply(len) # Count number of Characters
            df[cols + '_num_words' + suffix_feature] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
            df[cols + '_num_unique_words' + suffix_feature] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
            df[cols + '_words_vs_unique' + suffix_feature] =   \
                    df[cols+'_num_unique_words'+ suffix_feature] /   \
                    df[cols+'_num_words'+ suffix_feature] * 100 # Count Unique Words
        print_memory()

        print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
        language_stop = set(stopwords.words(language))   
        tfidf_para = {
            "stop_words": language_stop,
            "analyzer": 'word',
            "token_pattern": r'\w{1,}',
            "sublinear_tf": True,
            "dtype": np.float32,
            "norm": 'l2',
            #"min_df":5,
            #"max_df":.9,
            "smooth_idf":False
        }
        if max_features > 0:
            vectorizer = FeatureUnion([
                    ('description',TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=max_features,
                        **tfidf_para,
                        preprocessor=get_col('description'))),
                    ('text_feat',CountVectorizer(
                        ngram_range=(1, 2),
                        preprocessor=get_col('text_feat'))),
                    ('title',TfidfVectorizer(
                        ngram_range=(1, 2),
                        **tfidf_para,
                        preprocessor=get_col('title')))
                ])  
        else:
            vectorizer = FeatureUnion([
                    ('description',TfidfVectorizer(
                        ngram_range=(1, 2),
                        **tfidf_para,
                        preprocessor=get_col('description'))),
                    ('text_feat',CountVectorizer(
                        ngram_range=(1, 2),
                        preprocessor=get_col('text_feat'))),
                    ('title',TfidfVectorizer(
                        ngram_range=(1, 2),
                        **tfidf_para,
                        preprocessor=get_col('title')))
                ])                               

        vectorizer.fit(df.to_dict('records'))
        ready_df = vectorizer.transform(df.to_dict('records'))
        tfvocab = vectorizer.get_feature_names() 
        print_memory()

        # Drop Text Cols
        df.drop(textfeats, axis=1,inplace=True)

        print("Modeling Stage")
        # Combine Dense Features with Sparse Text Bag of Words Features
        # ready_df = ready_df.astype(np.float32)
        X = hstack([csr_matrix(df.values),ready_df])
        for shape in [X]:
            print("{} Rows and {} Cols".format(*shape.shape))
        print_memory()
                      
        save_file(df, filename_len, ext)

        print('>> saving to', filename)
        with open(filename, "wb") as f:
            pickle.dump((ready_df,tfvocab), f)

        del ready_df, tfvocab; gc.collect()
        
        with open(filename, "rb") as f:
            ready_df, tfvocab = pickle.load(f)             
    return df, ready_df        



CATEGORICAL = [
    'user_id', 'region', 'city', 'parent_category_name',
    'category_name', 'user_type', 'image_top_1'
]

def create_label_encode(df, todir, ext):
    print('\n>> create label encoding')
    filename = todir + 'cat_encode' + ext
    if os.path.exists(filename):
        print ('done already...')
        gp = load_file(filename, ext)
    else:
        gp = pd.DataFrame()
        lbl = preprocessing.LabelEncoder()
        for feature in CATEGORICAL:
            if feature in df:
                encoded_feature = map_key(feature) + '_encoded'
                gp[encoded_feature] = lbl.fit_transform(df[feature])
        save_file(df=gp, filename=filename, ext=ext)  
    return gp  

def create_time(df, todir, ext):
    print('\n>> extract time')
    filename = todir + 'time' + ext
    if os.path.exists(filename):
        print ('done already...')
        gp = load_file(filename, ext)
    else:
        gp = pd.DataFrame()
        gp['week'] = pd.to_datetime(df.activation_date).dt.week.astype('uint8')
        gp['weekday'] = pd.to_datetime(df.activation_date).dt.weekday.astype('uint8')
        gp['day'] = pd.to_datetime(df.activation_date).dt.day.astype('uint8')
        # print('\n >> saving to', filename)
        save_file(df=gp, filename=filename, ext=ext)  
    return gp      