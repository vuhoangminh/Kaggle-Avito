import numpy as np
import argparse
from path import Path
import pickle
import pandas as pd
import os, psutil
import time

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from utils.score_utils import mean_score, std_score

process = psutil.Process(os.getpid())

def print_debug(debug):
    if not debug:
        print('=======================================================================')
        print('process on server...')
        print('=======================================================================')
    else:
        print('=======================================================================')
        print('for testing only...')
        print('=======================================================================')

def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', 
            round(process.memory_info().rss/(2**30),2), ' GB\n')

def save_pickle(df, filename):
    print('>> saving to {}'.format(filename))
    with open(filename, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL) 

parser = argparse.ArgumentParser(description='Evaluate NIMA')
parser.add_argument('-resize', type=str, default='true',
                    help='Resize images to 224x224 before scoring')
parser.add_argument('-b', type=int, default=2,
                    choices=[2,0])

args = parser.parse_args()
DEBUG = args.b

if DEBUG:
    train_image_dir = '../input/image/images_train/'
    test_image_dir = '../input/image/images_test/'
    train_filename = '../processed_features_debug2/train_inceptionnet_score.pickle'
    test_filename = '../processed_features_debug2/test_inceptionnet_score.pickle'
else:
    train_image_dir = '../input/image/train_jpg/'
    test_image_dir = '../input/image/test_jpg/'  
    train_filename = '../processed_features/train_inceptionnet_score.pickle'
    test_filename = '../processed_features/test_inceptionnet_score.pickle'          

target_size = (224, 224)  

def main():

    print_debug(DEBUG)
    # for dataset in [train_image_dir, test_image_dir]:
    for dataset in [test_image_dir, train_image_dir]:        
        print('========================================================================')
        print('INCEPTIONNET')
        print('PROCESSING', dataset)
        print('========================================================================')

        df = pd.DataFrame(columns=['item_id','mean','std'])

        print("Loading images from directory : ", dataset)
        imgs = Path(dataset).files('*.png')
        imgs += Path(dataset).files('*.jpg')
        imgs += Path(dataset).files('*.jpeg')

        N = len(imgs)
        i = 0
        # with tf.device("CPU:0"):
        with tf.device("/device:GPU:0"):
            print('>> init')
            base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
            x = Dropout(0.75)(base_model.output)
            x = Dense(10, activation='softmax')(x)

            print('>> load weights')
            model = Model(base_model.input, x)
            model.load_weights('weights/inception_resnet_weights.h5')

            score_list = []

            df_temp = pd.DataFrame()
            if DEBUG: STEP=3
            else: STEP=1000
            if dataset == train_image_dir: todir = train_filename
            else: todir = test_filename  

            for img_path in imgs:
                if i%STEP==0:
                    end_step = time.time()
                    print('----------------------------')
                    print('{}/{}'.format(i,N))  
                    if i>0: 
                        print('time elapse:', end_step-start_step)
                        df = pd.concat( [df, df_temp], axis=0)                     
                        save_pickle(df, todir)
                        df_temp = pd.DataFrame()
                    start_step = time.time()                        
                if DEBUG: print("\n>> Evaluating : ", img_path)

                img = load_img(img_path, target_size=target_size)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                x = preprocess_input(x)

                scores = model.predict(x, batch_size=1, verbose=0)[0]

                mean = mean_score(scores)
                std = std_score(scores)

                file_name = Path(img_path).name.lower()
                score_list.append((file_name, mean))
            
                if DEBUG: print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))

                filename_w_ext = os.path.basename(img_path)
                filename, file_extension = os.path.splitext(filename_w_ext)

                temp = pd.DataFrame({'item_id': [filename],
                                    'mean': [mean],
                                    'std': [std]})
                if DEBUG: print(temp)

                df = pd.concat( [df, temp], axis=0) 

                i = i + 1

        df = pd.concat([df, df_temp], axis=0)
        df = df.reset_index(drop=True)
        print (df)               
        save_pickle(df, todir)

if __name__ == '__main__':
    main()    