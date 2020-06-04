"""
Configuration
"""
import sys
import os
import pickle
envipath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(envipath)
sys.path.append(envipath)

class TCconfig(object):

    datapath = 'Data/labels.xlsx'
    embedding_dim = 300  # embedding_dim
    num_words = 10000  # wiki high frequency words
    maxlenth = 50
    try:
        classes_pickle_file = open(os.path.join(envipath, 'Pickle/classes_pickle_file.pkl'),'rb')
        classnames = pickle.load(classes_pickle_file)
        print('classnames:',classnames)
    except FileNotFoundError:
        pass
