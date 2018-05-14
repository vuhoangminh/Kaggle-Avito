# PREDICTORS = [
#     'cn_encoded', 'cty_encoded', 'img1_encoded', 'pcn_encoded',
#     'reg_encoded', 'uid_encoded', 'uty_encoded',

#     # 'category_name', 'city', 'region', 'parent_category_name', 
#     # 'user_type', 'item_seq_number', 'image_top_1',
#     'day', 'week', 'weekday', 
#     # 'user_id', 
    
#     'price',

#     # kernel's
#     'description_num_chars',
#     # 'description_num_chars_en',
#     'description_num_unique_words',
#     # 'description_num_unique_words_en',
#     'description_num_words',
#     # 'description_num_words_en',
#     'description_words_vs_unique',
#     # 'description_words_vs_unique_en',
#     'text_feat_num_chars',
#     # 'text_feat_num_chars_en',
#     'text_feat_num_unique_words',
#     # 'text_feat_num_unique_words_en',
#     'text_feat_num_words',
#     # 'text_feat_num_words_en',
#     'text_feat_words_vs_unique',
#     # 'text_feat_words_vs_unique_en',
#     'title_num_chars',
#     # 'title_num_chars_en',
#     'title_num_unique_words',
#     # 'title_num_unique_words_en',
#     'title_num_words',
#     # 'title_num_words_en',
#     'title_words_vs_unique',
#     # 'title_words_vs_unique_en',

#     # minh's features
#     'cn_mean_dp',
#     'cty_mean_dp',
#     'cty_pcn_cn_mean_dp',
#     'cty_pcn_mean_dp',
#     'cty_uty_mean_dp',
#     'cty_uty_pcn_cn_mean_dp',
#     'cty_uty_pcn_mean_dp',
#     'pcn_cn_mean_dp',
#     'pcn_mean_dp',
#     # 'price',
#     'reg_mean_dp',
#     'reg_pcn_cn_mean_dp',
#     'reg_pcn_mean_dp',
#     'reg_uty_mean_dp',
#     'reg_uty_pcn_cn_mean_dp',
#     'reg_uty_pcn_mean_dp',
#     'uid_cn_mean_dp',
#     'uid_mean_dp',
#     'uid_pcn_cn_mean_dp',
#     'uid_pcn_mean_dp',
#     'uty_cn_mean_dp',
#     'uty_mean_dp',
#     'uty_pcn_cn_mean_dp',
#     'uty_pcn_mean_dp'
# ]        

PREDICTORS_BASED = [
    'time', 'cat_encode', 'len_feature_kernel',
] 

PREDICTORS_OVERFIT = [
    # overfit
    'uid_cn_mean_dp',
    'uid_mean_dp',
    'uid_pcn_cn_mean_dp',
    'uid_pcn_mean_dp',
] 

PREDICTORS_GOOD = [
    'cty_uty_pcn_cn_mean_dp',   #151
    'reg_uty_pcn_cn_mean_dp',   #63
    'cty_pcn_cn_mean_dp',       #36
    'uty_cn_mean_dp',           #29
    'cty_uty_pcn_mean_dp',      #27
    'cty_pcn_mean_dp',          #15
    'reg_pcn_mean_dp',          #15
    'reg_uty_pcn_mean_dp',      #15
    'cty_mean_dp',               #14
    'cty_uty_mean_dp',          #12
    'reg_pcn_cn_mean_dp',       #12
    'reg_uty_mean_dp',          #10
]

PREDICTORS_NOTCHECKED = [
    # option 1 low score
    'cn_mean_dp',
    'pcn_cn_mean_dp',
    'pcn_mean_dp',
    'reg_mean_dp',
    'uty_mean_dp',
    'uty_pcn_cn_mean_dp',
    'uty_pcn_mean_dp'
] 

PREDICTORS3 = [
    'time', 'cat_encode', 'len_feature_kernel',

    'cty_uty_pcn_cn_mean_dp',   #151
    'reg_uty_pcn_cn_mean_dp',   #63
    'cty_pcn_cn_mean_dp',       #36
    'uty_cn_mean_dp',           #29
    'cty_uty_pcn_mean_dp',      #27
    'cty_pcn_mean_dp',          #15
    'reg_pcn_mean_dp',          #15
    'reg_uty_pcn_mean_dp',      #15
    'cty_mean_dp'               #14
    'cty_uty_mean_dp',          #12
    'reg_pcn_cn_mean_dp',       #12
    'reg_uty_mean_dp',          #10

    ## option 1 low score
    # 'cn_mean_dp',
    # 'pcn_cn_mean_dp',
    # 'pcn_mean_dp',
    # 'reg_mean_dp',
    # 'uty_mean_dp',
    # 'uty_pcn_cn_mean_dp',
    # 'uty_pcn_mean_dp'

    ## overfit
    # 'uid_cn_mean_dp',
    # 'uid_mean_dp',
    # 'uid_pcn_cn_mean_dp',
    # 'uid_pcn_mean_dp',
] 

MINH_LIST_MEAN_DEAL_PROB =[
    ['user_id', 'deal_probability'],
    ['user_id', 'parent_category_name', 'deal_probability'],
    ['user_id', 'parent_category_name', 'category_name', 'deal_probability'],
    ['user_id', 'category_name', 'deal_probability'],

    ['user_type', 'deal_probability'],
    ['user_type', 'parent_category_name', 'deal_probability'],
    ['user_type', 'parent_category_name', 'category_name', 'deal_probability'],
    ['user_type', 'category_name', 'deal_probability'],

    ['region', 'deal_probability'],    
    ['region', 'parent_category_name', 'deal_probability'],
    ['region', 'parent_category_name', 'category_name', 'deal_probability'],  
    ['region', 'user_type', 'deal_probability'],
    ['region', 'user_type', 'parent_category_name', 'deal_probability'],
    ['region', 'user_type', 'parent_category_name', 'category_name', 'deal_probability'],  

    ['city', 'deal_probability'],
    ['city', 'parent_category_name', 'deal_probability'],
    ['city', 'parent_category_name', 'category_name', 'deal_probability'], 
    ['city', 'user_type', 'deal_probability'],
    ['city', 'user_type', 'parent_category_name', 'deal_probability'],
    ['city', 'user_type', 'parent_category_name', 'category_name', 'deal_probability'],          

    ['parent_category_name', 'deal_probability'],
    ['parent_category_name', 'category_name', 'deal_probability'],
    ['category_name', 'deal_probability'],      
]

MINH_LIST_MEAN_PRICE =[
    ['user_id', 'price'],
    ['user_id', 'parent_category_name', 'price'],
    ['user_id', 'parent_category_name', 'category_name', 'price'],
    ['user_id', 'category_name', 'price'],

    ['user_type', 'price'],
    ['user_type', 'parent_category_name', 'price'],
    ['user_type', 'parent_category_name', 'category_name', 'price'],
    ['user_type', 'category_name', 'price'],

    ['region', 'price'],    
    ['region', 'parent_category_name', 'price'],
    ['region', 'parent_category_name', 'category_name', 'price'],  
    ['region', 'user_type', 'price'],
    ['region', 'user_type', 'parent_category_name', 'price'],
    ['region', 'user_type', 'parent_category_name', 'category_name', 'price'],  

    ['city', 'price'],
    ['city', 'parent_category_name', 'price'],
    ['city', 'parent_category_name', 'category_name', 'price'], 
    ['city', 'user_type', 'price'],
    ['city', 'user_type', 'parent_category_name', 'price'],
    ['city', 'user_type', 'parent_category_name', 'category_name', 'price'],          

    ['parent_category_name', 'price'],
    ['parent_category_name', 'category_name', 'price'],
    ['category_name', 'price'],      
]

MINH_LIST_VAR_DEAL_PROB =[
    ['user_type', 'deal_probability'],
    ['user_type', 'parent_category_name', 'deal_probability'],
    ['user_type', 'parent_category_name', 'category_name', 'deal_probability'],
    ['user_type', 'category_name', 'deal_probability'],

    ['region', 'deal_probability'],    
    ['region', 'parent_category_name', 'deal_probability'],
    ['region', 'parent_category_name', 'category_name', 'deal_probability'],  
    ['region', 'user_type', 'deal_probability'],
    ['region', 'user_type', 'parent_category_name', 'deal_probability'],
    ['region', 'user_type', 'parent_category_name', 'category_name', 'deal_probability'],  

    ['city', 'deal_probability'],
    ['city', 'parent_category_name', 'deal_probability'],
    ['city', 'parent_category_name', 'category_name', 'deal_probability'], 
    ['city', 'user_type', 'deal_probability'],
    ['city', 'user_type', 'parent_category_name', 'deal_probability'],
    ['city', 'user_type', 'parent_category_name', 'category_name', 'deal_probability'],          

    ['parent_category_name', 'deal_probability'],
    ['parent_category_name', 'category_name', 'deal_probability'],
    ['category_name', 'deal_probability'],      
]

MINH_LIST_VAR_PRICE =[
    ['user_type', 'price'],
    ['user_type', 'parent_category_name', 'price'],
    ['user_type', 'parent_category_name', 'category_name', 'price'],
    ['user_type', 'category_name', 'price'],

    ['region', 'price'],    
    ['region', 'parent_category_name', 'price'],
    ['region', 'parent_category_name', 'category_name', 'price'],  
    ['region', 'user_type', 'price'],
    ['region', 'user_type', 'parent_category_name', 'price'],
    ['region', 'user_type', 'parent_category_name', 'category_name', 'price'],  

    ['city', 'price'],
    ['city', 'parent_category_name', 'price'],
    ['city', 'parent_category_name', 'category_name', 'price'], 
    ['city', 'user_type', 'price'],
    ['city', 'user_type', 'parent_category_name', 'price'],
    ['city', 'user_type', 'parent_category_name', 'category_name', 'price'],          

    ['parent_category_name', 'price'],
    ['parent_category_name', 'category_name', 'price'],
    ['category_name', 'price'],      
]

# count same product? should process nlp
MINH_LIST_COUNT =[
    # ['city', 'category_name'],
    # ['region', 'category_name'],
    ['user_id'],
    ['user_id', ],
]

MINH_LIST_UNIQUE = [
    ['city', 'category_name'],
]

MINH_LIST_FREQ = [
    ['user_id'],

]


# 'category_name', 'city', 'region', 'parent_category_name', 
# 'user_type', 'item_seq_number', 'image_top_1',
# 'day', 'week', 'weekday', 
# 'user_id', 