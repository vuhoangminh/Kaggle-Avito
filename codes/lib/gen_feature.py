import os, sys, inspect
import pandas as pd
import gc
from .read_write_file import save_file, load_file



def generate_groupby_by_type_and_columns(df, selcols, apply_type, todir, ext):      
    feature_name = ''
    for i in range(len(selcols)-1):
        feature_name = feature_name + selcols[i] + '_'
    feature_name = feature_name + apply_type + '_' + selcols[len(selcols)-1]
    print('>> doing feature:', feature_name)
    
    filename = todir + feature_name + ext

    if os.path.exists(filename):
        print ('done already...')
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
        print('>> saving to', filename)
        save_file(df=col_extracted,filename=filename,ext=ext)                      
    return feature_name 

def create_time(df, todir, ext):
    print('>> extract time')
    filename = todir + 'time' + ext
    if os.path.exists(filename):
        print ('done already...')
    else:
        gp = pd.DataFrame()
        gp['week'] = pd.to_datetime(df.activation_date).dt.week.astype('uint8')
        gp['weekday'] = pd.to_datetime(df.activation_date).dt.weekday.astype('uint8')
        gp['day'] = pd.to_datetime(df.activation_date).dt.day.astype('uint8')
        save_file(df=gp, filename=filename, ext=ext)        


if __name__ == '__main__':
    _test()