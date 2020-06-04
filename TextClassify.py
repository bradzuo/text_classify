"""
author: Pactera Risk AI
time:2020.1.7
model:BiLSTM
Train for model
"""

# coding:utf-8
import numpy as np
import re
import jieba
import os,sys
envipath = os.path.dirname(os.getcwd())
sys.path.append(envipath)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LSTM,Embedding,Bidirectional,Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from tensorflow.python.keras.utils import to_categorical
import pickle
from config.config import TCconfig
import pandas as pd
from Utils.LoadData import load_data
from Utils.Log import Logger
logger = Logger().logger

class BiLstmTextClassify(object):

    def __init__(self):

        self.origh_data_path = os.path.join(envipath, 'Data/TRAINs.csv')
        self.wiki_pickle_path = os.path.join(envipath, 'wiki_matrix/cn_model.pickle')
        self.maxlenth = TCconfig.maxlenth
        load_data()
        logger.info('start load data')

    def generate_classes(self,labels):
        """
        # 动态生成模型的文本类型
        :param labels:
        :return:
        """
        labels_dict = {}
        all_labels = list(set(labels))
        for label_index,label_value in enumerate(all_labels):
            labels_dict[label_value] = label_index
        return labels_dict

    def data_process(self,):
        """
        @ load origin data and deal with labels
        :return:
        """
        df = pd.read_csv(self.origh_data_path, index_col=0)
        texts = df['ARTI_TITLE']
        labels = df['TAG_NAME'].tolist()
        self.classnames = self.generate_classes(labels=labels)
        classes_pickle_file = open(os.path.join(envipath, 'Pickle/classes_pickle_file.pkl'), 'wb')
        pickle.dump(self.classnames,classes_pickle_file)
        print('文本类型：',self.classnames)
        print('文本类型数量：',self.classnames.values().__len__())
        targets = [self.classnames[item] for item in labels]
        targets = np.array(targets)

        logger.info('num of origin texts：', len(texts), type(targets))
        logger.info('num of origin labels：', targets.shape[0])
        return texts, targets

    def load_wiki(self,text_orig):
        """
        @ loading wiki_matrix,which is based on to turn chinese to numbers
        :param text_orig:
        :return:
        """
        train_tokens = []
        with open(os.path.join(envipath,'Pickle/words_and_index_dict.pickle'), 'rb') as fr:
            cn_model = pickle.load(fr)
        for text in text_orig:
            text = re.sub('[\s+\.\!\/_,$%^*(+\"\']+|[+——！-．（），:。？、；：~@#%……&*()]+', '', text)
            cut = jieba.cut(text)
            cut_list = [item for item in cut]
            for i, word in enumerate(cut_list):
                try:
                    cut_list[i] = cn_model[word]
                except KeyError:
                    cut_list[i] = TCconfig.num_words + 1
            train_tokens.append(cut_list)

        return train_tokens, cn_model

    def data_padding(self,train_tokens, max_tokens):
        """
        @ padding ,in order to make sure the lenth of input is the same
        :param train_tokens:
        :param max_tokens:
        :return:
        """
        # number of sequences stands for how many words in a sentence,every word will be replace by a one_hot
        train_pad = pad_sequences(train_tokens, maxlen=max_tokens, padding='pre', truncating='pre')
        # 超出词汇表的都转换为0
        train_pad[train_pad >= TCconfig.num_words] = 0
        logger.info('padding is done')
        return train_pad

    def init_embedding(self):
        """
        @ init embedding_matrix
        :param cn_model:
        :return:
        """
        with open(os.path.join(envipath,'Pickle/init_embed.pickle'), 'rb') as fr:
            embedding_matrics = pickle.load(fr)
        return embedding_matrics

    def split_data(self,train_pad, train_target):
        """
        @ split data to train and test
        :param train_pad:
        :param train_target:
        :return:
        """

        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(train_pad, train_target, test_size=0.2, random_state=12)
        logger.info('trainsize', X_train.shape, 'testsize', Y_train.shape)
        # print('trainsize', X_train.shape, 'testsize', Y_train.shape)
        # mutliple_class need to use to_categorical to make y one_hot
        # need to define num_classes,or it will define num_classes as the max_number_class exsists
        Y_train = to_categorical(Y_train,num_classes=self.classnames.values().__len__())
        # print('Y_train.shape:',Y_train.shape)
        return X_train, X_test, Y_train, Y_test

    def make_model(self,embedding_matrics, max_tokens, X_train, X_test, Y_train, Y_test):

        """
        @ set up model
        :param embedding_matrics:
        :param max_tokens:
        :param X_train:
        :param X_test:
        :param Y_train:
        :param Y_test:
        :return:
        """
        model = Sequential()
        # embedding层的作用其实就是查表，从而将高维稀疏矩阵转变为低纬稠密矩阵
        model.add(Embedding(TCconfig.num_words, TCconfig.embedding_dim, weights=[embedding_matrics], input_length=max_tokens,
                      trainable=False))
        # model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
        # model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(units=32, return_sequences=False)))
        model.summary()
        print('Y_train.shape[1]:',Y_train.shape[1])
        model.add(Dense(Y_train.shape[1], activation='softmax'))
        optimizer = Adam(lr=1e-3)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()

        # defin callbacks
        path_checkpoint = os.path.join(envipath, 'Checkpoints/news_classifier_checkpoint.h5')
        checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=False,
                                     save_best_only=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)
        tensorboard = TensorBoard(log_dir=os.path.join(envipath, 'Log'))
        callbacks = [earlystopping, lr_reduction, checkpoint, tensorboard]
        print('start to train........')
        logger.info('start to train........')
        # train
        model.fit(X_train, Y_train,
                  validation_split=0.1,
                  epochs=20,
                  batch_size=64,
                  shuffle=True,
                  callbacks=callbacks)

        # print('Y_test:',Y_test)
        Y_test = to_categorical(Y_test,num_classes=self.classnames.values().__len__())
        result = model.evaluate(X_test, Y_test)
        logger.info('accuracy:{}'.format(result[1]))
        print('accuracy:{}'.format(result[1]))
        # save model
        print('saving model is done...')
        logger.info('saving model is done')

    def Begin(self):
        text_orig, train_target = self.data_process() # load data
        train_tokens, cn_model = self.load_wiki(text_orig) #　load wiki
        train_pad = self.data_padding(train_tokens,self.maxlenth)  # padding
        embedding_matrics = self.init_embedding() # set up model
        #TODO: data generator
        X_train, X_test, Y_train, Y_test = self.split_data(train_pad, train_target) # split data to train and test
        self.make_model(embedding_matrics,self.maxlenth,X_train,X_test,Y_train,Y_test) #　train and save

if __name__ == '__main__':
    BiLstmTextClassify().Begin()