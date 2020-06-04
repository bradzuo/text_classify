"""
author: Pactera Risk AI
time:2020.1.7
model:BiLSTM
Model to predict
"""

"""
predict text which label is 
"""
import re
import os,sys
import jieba
#from gensim.models import KeyedVectors
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
#from keras.models import model_from_json
import os,sys
from config.config import TCconfig
from tensorflow.python.keras.models import load_model
import pickle
from Utils.Log import Logger
logger = Logger().logger
logger.info('start to predict')

try:
    text = sys.argv[1]
except:
    text = '文思海辉被中国电子收购'

def load_wiki():
    """
    @ load wiki matrix
    :return:
    """
    wiki_pickle_path = os.path.join(os.getcwd(),'Pickle/words_and_index_dict.pickle')
    with open(wiki_pickle_path,'rb') as fr:
        cn_model = pickle.load(fr)
    return cn_model

def load_trained_model():
    '''
    @ load trained model
    :return:
    '''

    model_path = os.path.join(os.getcwd(),'Checkpoints/news_classifier_checkpoint.h5')
    model = load_model(model_path)
    return model

class StartToPredict(object):

    def __init__(self):
        self.cn_model = load_wiki()
        self.model = load_trained_model()
        self.num_words = TCconfig.num_words
        self.classnames = TCconfig.classnames
        self.maxlenth = TCconfig.maxlenth
        self.text = text

    def start_to_predict(self):

        logger.info('input text')
        text=re.sub('[\s+\.\!\/_,$%^*(+\"\']+|[+——！-．（），:。？、；：~@#%……&*()]+','',self.text)
        text_cut=jieba.cut(text)
        text_cut=[item for item in text_cut]
        for i,word in enumerate(text_cut):
            try:
                text_cut[i] = self.cn_model[word]
            except KeyError:
                text_cut[i] = TCconfig.num_words + 1
        text_tokens=text_cut

        test_pad=pad_sequences([text_tokens],maxlen=self.maxlenth,padding='pre',truncating='pre')
        test_pad[test_pad> self.num_words]=0
        result = self.model.predict(test_pad)

        text_class=[k for k,v in self.classnames.items() if v==np.argmax(result,axis=1)[0]][0]
        label_results = '{"label":"' + text_class  + '"}'
        print(label_results)
        return label_results

if __name__ == '__main__':

    StartToPredict().start_to_predict()
