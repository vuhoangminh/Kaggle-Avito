import matplotlib
matplotlib.use('Agg')

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

cwd = os.getcwd()
print ('working dir', cwd)
if 'codes' not in cwd:
    default_path = cwd + '/codes/'
    os.chdir(default_path)

# print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 
from lib.read_write_file import save_csv, save_feather, save_file, save_pickle
from lib.read_write_file import load_csv, load_feather, load_pickle, read_train_test
from lib.prepare_training import get_text_matrix, read_processed_h5

import nltk
nltk.download('stopwords')

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

SEED = 1988

print("\nData Load Stage")

# storename = '../processed_features_debug2/train_debug2.h5'.format(DEBUG, 'train', DEBUG)

# training = load_feather('../input/debug2/train_debug2.feather')
# training.set_index('item_id', inplace=True)
# training['activation_date'] = pd.to_datetime(training['activation_date'])
# traindex = training.index
# testing = load_feather('../input/debug2/test_debug2.feather')
# testing.set_index('item_id', inplace=True)
# testing['activation_date'] = pd.to_datetime(testing['activation_date'])
# testdex = testing.index






target = ['deal_probability',]
predictors = [
          'activation_date',
          'category_name',
          'image',
          'image_top_1',
          'item_id',
          'city',
          'description',
          'item_seq_number',
          'param_1',
          'param_2',
          'param_3',
          'parent_category_name',
          'price',
          'title',
          'user_id',
          'user_type',
          'region',
]
categorical = []

storename = storename = '../processed_features_debug{}/{}_debug{}.h5'.format(2, 'train', 2)
training = read_processed_h5(storename, predictors+target, categorical)
training.set_index('item_id', inplace=True)
training['activation_date'] = pd.to_datetime(training['activation_date'])
traindex = training.index

storename = storename = '../processed_features_debug{}/{}_debug{}.h5'.format(2, 'test', 2)
testing = read_processed_h5(storename, predictors, categorical)
testing.set_index('item_id', inplace=True)
testing['activation_date'] = pd.to_datetime(testing['activation_date'])
testdex = testing.index



y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print(df.info()); print(df.head())

print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-999,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

# Create Validation Index and Remove Dead Variables
training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
df.drop(["activation_date","image"],axis=1,inplace=True)

print("\nEncode Variables")
categorical = ["user_id","region","city","parent_category_name",
"category_name","item_seq_number","user_type","image_top_1"]
print("Encoding :",categorical)

print(df.info())

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))
    
print("\nText Features")


print(df.info())


# Feature Engineering 
df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']), 
    str(row['param_2']), 
    str(row['param_3'])]),axis=1) # Group Param Features

print (df[['text_feat', 'param_1', 'param_2', 'param_3']].head())
print (df[['text_feat', 'param_1', 'param_2', 'param_3']].tail())



df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

# Meta Text Features
textfeats = ["description","text_feat", "title"]
for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('nicapotato') # FILL NA
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
def get_col(col_name): return lambda x: x[col_name]
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

print(df.info())

start_vect=time.time()
vectorizer.fit(df.loc[traindex,:].to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

# Drop Text Cols
df.drop(textfeats, axis=1,inplace=True)

# Dense Features Correlation Matrix
f, ax = plt.subplots(figsize=[10,7])
sns.heatmap(pd.concat([df.loc[traindex,[x for x in df.columns if x not in categorical]], y], axis=1).corr(),
            annot=False, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="plasma",ax=ax, linewidths=.5)
ax.set_title("Dense Features Correlation Matrix")
plt.savefig('correlation_matrix.png')

print("Modeling Stage")
# Combine Dense Features with Sparse Text Bag of Words Features

print(df.info())
# print(ready_df.info())

print(df.head())

print(ready_df)

X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();

print("\nModeling Stage")

# Training and Validation Set
"""
Using Randomized train/valid split doesn't seem to generalize LB score, so I will try time cutoff
"""
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.10, random_state=SEED)

print(X.shape)

print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 15,
    # 'num_leaves': 31,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'learning_rate': 0.019,
    'verbose': 0
}  

# LGBM Dataset Formatting 
lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=tfvocab,
                categorical_feature = categorical)
lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab,
                categorical_feature = categorical)

# Go Go Go
modelstart = time.time()
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=16000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=200,
    verbose_eval=200
)

# # Feature Importance Plot
# f, ax = plt.subplots(figsize=[7,10])
# lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
# plt.title("Light GBM Feature Importance")
# plt.savefig('feature_import.png')

# print("Model Evaluation Stage")
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
# lgpred = lgb_clf.predict(testing)
# lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
# lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
# lgsub.to_csv("lgsub.csv",index=True,header=True)
# print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
# print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))