import numpy as np
import argparse
from path import Path
import pickle
import pandas as pd
import os, psutil

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from utils.nasnet import NASNetMobile, preprocess_input
from utils.score_utils import mean_score, std_score

process = psutil.Process(os.getpid())

from keras import backend as K
num_cores = 4
num_GPU = 1
num_CPU = 1

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

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
    train_filename = '../processed_features_debug2/train_nasnet_score.pickle'
    test_filename = '../processed_features_debug2/test_nasnet_score.pickle'
else:
    train_image_dir = '../input/image/train_jpg/'
    test_image_dir = '../input/image/test_jpg/'  
    train_filename = '../processed_features/train_nasnet_score.pickle'
    test_filename = '../processed_features/test_nasnet_score.pickle'          

target_size = (224, 224)  

def main():

    print_debug(DEBUG)
    for dataset in [train_image_dir, test_image_dir]:
        print('========================================================================')
        print('NASNET')
        print('PROCESSING', dataset)
        print('========================================================================')

        df = pd.DataFrame(columns=['item_id','mean','std'])

        print("Loading images from directory : ", dataset)
        imgs = Path(dataset).files('*.png')
        imgs += Path(dataset).files('*.jpg')
        imgs += Path(dataset).files('*.jpeg')

        with tf.device("CPU:0"):
            print('>> init')
            base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg', weights=None)
            x = Dropout(0.75)(base_model.output)
            x = Dense(10, activation='softmax')(x)

            print('>> load weights')
            model = Model(base_model.input, x)
            model.load_weights('weights/nasnet_weights.h5')

            score_list = []

            for img_path in imgs:
                print("\n>> Evaluating : ", img_path)

                img = load_img(img_path, target_size=target_size)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                x = preprocess_input(x)

                scores = model.predict(x, batch_size=1, verbose=0)[0]

                mean = mean_score(scores)
                std = std_score(scores)

                file_name = Path(img_path).name.lower()
                score_list.append((file_name, mean))
            
                print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
                print()

                filename_w_ext = os.path.basename(img_path)
                filename, file_extension = os.path.splitext(filename_w_ext)

                temp = pd.DataFrame({'item_id': [filename],
                                    'mean': [mean],
                                    'std': [std]})
                if DEBUG: print(temp)

                df = pd.concat( [df, temp], axis=0) 

        df = df.reset_index(drop=True)
        print (df)   

        if dataset == train_image_dir:
            todir = train_filename
        else:
            todir = test_filename    
                    
        save_pickle(df, train_filename)

if __name__ == '__main__':
    main()    
    