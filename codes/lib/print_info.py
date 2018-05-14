import os, psutil

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

def print_doing(str):
    print('\n>>', str)    

def print_doing_in_task(str):
    print('>>', str)    

def print_header(str):
    print('\n--------------------------------------------------------------------') 
    print(str)
    print('--------------------------------------------------------------------')     
    