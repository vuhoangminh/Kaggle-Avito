# N = 45
import math

def build_dict():
    RAMSIZE = 100
    RATIO = 10/9
    N = math.floor(RAMSIZE/RATIO)
    # N = 9
    print(N)
    FRM = 0
    STEP = 34

    str = ''

    for i in range (N):
        temp = 'nohup python -u build_dict_fast.py -f {} -s {} & '.format(FRM+i*STEP, STEP)
        str = str + temp

    file = open('bash.txt','w') 
    file.write(str) 
    print (str)    

def translate_textblob():
    str = ''
    for dataset in ['train_translated', 'train_active_translated', 'test_translated',
                'test_active_translated']:
        for debug in [2,1,0]:
            temp = 'nohup python -u translate_textblob.py -b {} -d {} & '.format(debug, dataset)
            str = str + temp           

    file = open('bash.txt','w') 
    file.write(str) 
    print (str)         

translate_textblob()    