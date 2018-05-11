# N = 45
import math
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
