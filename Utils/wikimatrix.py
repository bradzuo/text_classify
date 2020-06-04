"""
load wiki matrix
"""
import sys,os
import numpy as np
import pickle
envipath = os.path.dirname(os.getcwd())
sys.path.append(envipath)

num_words = 10000
embedding_dim = 300

def save_pickle(pickle_path,wcont):
    """
    @ save to pickle
    :return:
    """
    with open(pickle_path, 'wb') as fw:
        pickle.dump(wcont, fw)

def load_word_and_index():
    """
    @ load wiki words and index
    :return:
    """
    with open(os.path.join(envipath,'wiki_matrix/cn_model.pickle'), 'rb') as fr:
        cn_model = pickle.load(fr)
    words_and_index_dict = dict(zip([cn_model.index2word[i] for i in range(num_words)],np.arange(num_words)))
    dict_path = os.path.join(envipath,'Pickle/words_and_index_dict.pickle')
    # save to pickle
    save_pickle(dict_path, words_and_index_dict)
    return cn_model

def init_embedding(cn_model):
    """
    @ init embedding_matrix
    :param cn_model:
    :return:
    """
    embedding_matrics = np.zeros((num_words, embedding_dim))
    for i in range(num_words):
        embedding_matrics[i, :] = cn_model[cn_model.index2word[i]]
    embedding_matrics = embedding_matrics.astype('float32')
    embed_path = os.path.join(envipath,'Pickle/init_embed.pickle')
    # save to pickle
    save_pickle(embed_path, embedding_matrics)

if __name__ == '__main__':

    cn_model = load_word_and_index()
    init_embedding(cn_model)