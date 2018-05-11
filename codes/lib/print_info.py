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
    print('Total memory in use ' + print_string + ': ', round(process.memory_info().rss/(2**30),2), ' GB')