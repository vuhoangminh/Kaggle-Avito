import os, sys, inspect
import pandas as pd
import gc
from .read_write_file import save_file, load_file
from .configs import NAMEMAP_DICT

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
import nltk
nltk.download('stopwords')

# NAMEMAP_DICT = configs.NAMEMAP_DICT 

def map_key(key):
    if key in NAMEMAP_DICT:
        return NAMEMAP_DICT[key]
    else:
        return key

def generate_groupby_by_type_and_columns(df, selcols, apply_type, todir, ext):      
    feature_name = ''
    for i in range(len(selcols)-1):
        feature_name = feature_name + map_key(selcols[i]) + '_'
    feature_name = feature_name + apply_type + '_' + map_key(selcols[len(selcols)-1])
    print('>> doing feature:', feature_name)
    
    filename = todir + feature_name + ext

    if os.path.exists(filename):
        print ('done already...')
        col_extracted = load_file(filename, ext)
    else:
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
        if apply_type != 'cumcount':
            df = df.merge(col_temp, on=selcols[0:len(selcols)-1], how='left')
            del col_temp; gc.collect()
            col_extracted[feature_name] = df[feature_name]
        else:
            col_extracted[feature_name] = col_temp.values
            del col_temp; gc.collect()
        # print('>> saving to', filename)
        save_file(df=col_extracted,filename=filename,ext=ext)                      
    return col_extracted 

def create_time(df, todir, ext):
    print('>> extract time')
    filename = todir + 'time' + ext
    if os.path.exists(filename):
        print ('done already...')
        gp = load_file(filename, ext)
    else:
        gp = pd.DataFrame()
        gp['week'] = pd.to_datetime(df.activation_date).dt.week.astype('uint8')
        gp['weekday'] = pd.to_datetime(df.activation_date).dt.weekday.astype('uint8')
        gp['day'] = pd.to_datetime(df.activation_date).dt.day.astype('uint8')
        # print('>> saving to', filename)
        save_file(df=gp, filename=filename, ext=ext)  
    return gp              

def measure_length(df, selcols, todir, ext):
    print('>> extract len of', selcols)
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
        # print('>> saving to', filename)
        save_file(df=gp, filename=filename, ext=ext)  
    return gp.astype('int')

def get_col(col): 
    return lambda x: x[col]

def create_text_feature (df, todir, ext):
    print('\n>> doing Text Features')
    filename = todir + 'text_feature_kernel' + ext
    if os.path.exists(filename):
        print ('done already...')
        gp = load_file(filename, ext)
    else:
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
            df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters
            df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
            df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
            df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
        print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
        russian_stop = set(stopwords.words('russian'))   
        tfidf_para = {
            "stop_words": russian_stop,
            "analyzer": 'word',
            "token_pattern": r'\w{1,}',
            "sublinear_tf": True,
            "dtype": np.float32,
            "norm": 'l2',
            #"min_df":5,
            #"max_df":.9,
            "smooth_idf":False
        }
        vectorizer = FeatureUnion([
                ('description',TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=18000,
                    **tfidf_para,
                    preprocessor=get_col('description'))),
                ('text_feat',CountVectorizer(
                    ngram_range=(1, 2),
                    #max_features=7000,
                    preprocessor=get_col('text_feat'))),
                ('title',TfidfVectorizer(
                    ngram_range=(1, 2),
                    **tfidf_para,
                    #max_features=7000,
                    preprocessor=get_col('title')))
            ])   

        vectorizer.fit(df.to_dict('records'))
        ready_df = vectorizer.transform(df.to_dict('records'))
        tfvocab = vectorizer.get_feature_names() 

        print("Modeling Stage")
        # Combine Dense Features with Sparse Text Bag of Words Features
        X = hstack([csr_matrix(df.values),ready_df])
        tfvocab = df.columns.tolist() + tfvocab
        for shape in [X]:
            print("{} Rows and {} Cols".format(*shape.shape))
        print("Feature Names Length: ",len(tfvocab))
        del df; gc.collect()