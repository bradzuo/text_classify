"""
Load Origin Labels Data
"""
# coding:utf-8
import numpy as np
import pandas as pd
import sys,os
envipath = os.path.dirname(os.getcwd())
sys.path.append(envipath)
from config.config import TCconfig
import pickle

def to_dict(classnames):
    return dict(zip(classnames,np.arange(len(classnames))))

def load_data():
    """
    @ load origin data and save the labels
    :return:
    """
    data = pd.read_excel(os.path.join(envipath,TCconfig.datapath))
    TRAINs = data[['ARTI_TITLE','TAG_NAME']]
    TRAINs.to_csv(os.path.join(envipath,'Data/TRAINs.csv'))

if __name__ == '__main__':
    load_data()