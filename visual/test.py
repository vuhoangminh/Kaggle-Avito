import pickle
import pandas as pd


# filename = '../processed_features_debug2/train_nasnet_score.pickle'
# df = pickle.load(open(filename, "rb" ))
# print(filename)
# print(df)

# filename = '../processed_features_debug2/test_nasnet_score.pickle'
# df = pickle.load(open(filename, "rb" ))
# print(filename)
# print(df)

# filename = '../processed_features_debug2/train_mobilenet_score.pickle'
# df = pickle.load(open(filename, "rb" ))
# print(filename)
# print(df)

# filename = '../processed_features_debug2/test_mobilenet_score.pickle'
# df = pickle.load(open(filename, "rb" ))
# print(filename)
# print(df)

# filename = '../processed_features_debug2/train_inceptionnet_score.pickle'
# df = pickle.load(open(filename, "rb" ))
# print(filename)
# print(df)

# filename = '../processed_features_debug2/test_inceptionnet_score.pickle'
# df = pickle.load(open(filename, "rb" ))
# print(filename)
# print(df)

print('------------------------------------')
filename = 'test_nasnet_score.pickle'
df = pickle.load(open(filename, "rb" ))
print(filename)
print(df)
print(df.info())
print(df.describe())

print('------------------------------------')
filename = 'test_mobilenet_score.pickle'
df = pickle.load(open(filename, "rb" ))
print(filename)
print(df)
print(df.info())
print(df.describe())

print('------------------------------------')
filename = 'test_inceptionnet_score.pickle'
df = pickle.load(open(filename, "rb" ))
print(filename)
print(df)
print(df.info())
print(df.describe())