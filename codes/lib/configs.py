SEED = 1988

DATATYPE_DICT = {
    'count'     : 'uint32',
    'nunique'   : 'uint32',
    'cumcount'  : 'uint32',
    'var'       : 'float32',
    'std'       : 'float32',
    'confRate'  : 'float32',
    'nextclick' : 'int64',
    'mean'      : 'float32'
}

NAMEMAP_DICT = {
    'item_id'   : 'iid',
    'user_id'   : 'uid',
    'region'    : 'reg',
    'city'      : 'cty',
    'parent_category_name' : 'pcn',
    'category_name' :   'cn',
    'param_1'   : 'p1',
    'param_2'   : 'p2',
    'param_3'   : 'p3',
    'title'     : 'tit',
    'description'   : 'des',    
    'price'     : 'pr',
    'item_seq_number'   : 'isn',
    'activation_date'   : 'ad',
    'user_type' : 'uty',
    'image'     : 'img',
    'image_top_1'   : 'img1',
    'date_from' : 'dfr',
    'date_to'   : 'dto',
    'deal_probability'  : 'dp'
}  

SORT_ODER_DICT = {
    'item_id'   : 0,
    'user_id'   : 1,
    'region'    : 2,
    'city'      : 3,
    'parent_category_name' : 4,
    'category_name' :   5,
    'param_1'   : 6,
    'param_2'   : 7,
    'param_3'   : 8,
    'title'     : 9,
    'description'   : 10,
    'price'     : 11,
    'item_seq_number'   : 12,
    'activation_date'   : 13,
    'user_type' : 14,
    'image'     : 15,
    'image_top_1'   : 16,
    'date_from' : 17,
    'date_to'   : 18,
    'deal_probability'  : 19,
}  

DATATYPE_DICT = {
    'count'     : 'uint32',
    'nunique'   : 'uint32',
    'cumcount'  : 'uint32',
    'var'       : 'float32',
    'std'       : 'float32',
    'confRate'  : 'uint32',
    'nextclick' : 'int',
    'mean'      : 'float32'
}

DATATYPE_LIST = {
    'item_id'           : 'str',
    'user_id'           : 'str',
    'region'            : 'str',
    'city'              : 'str',
    'parent_category_name' : 'str',
    'category_name'     :   'str',
    'param_1'           : 'str',
    'param_2'           : 'str',
    'param_3'           : 'str',
    'title'             : 'str',
    'description'       : 'str',    
    # 'price'             : 'float',
    'item_seq_number'   : 'int',
    'activation_date'   : 'str',
    'user_type'         : 'str',
    'image'             : 'str',
    'image_top_1'       : 'str',
    'date_from'         : 'str',
    'date_to'           : 'str',
    'deal_probability'  : 'float'
} 

# TRANSLATE_LIST = {
#     'parent_category_name' : 'parent_category_name_en',
#     'category_name'     :   'category_name_en',
#     'param_1'           : 'param_1_en',
#     'param_2'           : 'param_2_en',
#     'param_3'           : 'param_3_en',
#     'title'             : 'title_en',
#     'description'       : 'description_en',     
# }

TRANSLATE_LIST = {
    'param_1'           : 'param_1_en',
    'param_2'           : 'param_2_en',
    'param_3'           : 'param_3_en',
    'title'             : 'title_en',
    'description'       : 'description_en',     
}
