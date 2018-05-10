N = 45
FRM = 140
STEP = 10

str = ''

for i in range (N):
    temp = 'python -u build_dict_fast.py -f {} -s {} & '.format(140+i*STEP, STEP)
    str = str + temp

print (str)    
