   
import pickle

dict2 = pickle.load(open("../dict/translated_train_title_239.pickle", "rb" ))

for key, value in dict2.items():
    print('from:', key)
    print('to:', value)

