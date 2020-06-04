"""
flask for model
"""

# -*-coding:utf-8-*-
import os,re,json,sys
# import keras
# keras.backend.clear_session()
from tensorflow import keras
import tensorflow as tf
import jieba
from flask import request, jsonify, render_template, Response,Flask
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.python.keras.models import load_model
import pickle
from config.config import TCconfig
from Utils.Log import Logger
logger = Logger().logger

app = Flask(__name__)
app.config['DEBUG'] = False
logger.info('start flask ...')

with open(os.path.join(os.getcwd(), 'Pickle/words_and_index_dict.pickle'), 'rb') as fr:
    cn_model = pickle.load(fr)
logger.info('load wiki ...')

model_path = os.path.join(os.getcwd(), 'Checkpoints/news_classifier_checkpoint.h5')
model = load_model(model_path)
logger.info('load trained model ...')
logger.info('init graph ...')

graph = tf.get_default_graph()
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras import backend as K


config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
K.set_session(sess)


session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)


def pre_deal(pred_text):
    """
    @ process input text
    :param pred_text:
    :return:
    """
    test_text = re.sub('[\s+\.\!\/_,$%^*(+\"\']+|[+——！-．（），:。？、；：~@#%……&*()]+', '', pred_text)
    text_cut = jieba.cut(test_text)
    text_cut = [item for item in text_cut]
    for i, word in enumerate(text_cut):
        try:
            text_cut[i] = cn_model[word]
        except KeyError:
            text_cut[i] = TCconfig.num_words + 1

    test_tokens = text_cut
    max_tokens = TCconfig.maxlenth
    test_pad = pad_sequences([test_tokens], maxlen=max_tokens, padding='pre', truncating='pre')
    test_pad[test_pad >= TCconfig.num_words] = 0
    return test_pad


def Response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route("/TextClassify", methods=["GET", "POST"])
def ZHClassify():
    global sess
    global graph
    try:
        content = ""
        if request.method == "POST":
            text = request.form.get("title")
        else:
            text = request.args.get("title")
    except Exception as e:
        logger.error(e)
    pred_text = re.sub(r"[^\w,，.。/?？\'\";；:：|{}\[\]\-【】《》<>=+(（)）*%￥#@！~·`$]", "", text)
    test_pad = pre_deal(pred_text)
    print('input text:', test_pad)
    logger.info('input text is ready ...')
    try:
        with graph.as_default():
            set_session(sess)
            result = model.predict(test_pad)
            type = np.argmax(result, axis=1)[0]
            print('result of label is predicted ', type)
            logger.info('result of label is predicted ', type)

            text_class = [k for k, v in TCconfig.classnames.items() if v == np.argmax(result, axis=1)[0]][0]
            content = json.dumps({'label':text_class},ensure_ascii=False)
            resp = Response_headers(content,)
    except Exception as e:
        logger.error(e)
        content = json.dumps({'error': e},ensure_ascii=False)
        resp = Response_headers(content)
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=False)
