import os, sys, inspect
import pandas as pd
import gc
from .read_write_file import save_file, load_file
from .configs import NAMEMAP_DICT

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