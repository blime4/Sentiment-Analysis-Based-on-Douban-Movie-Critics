### Model Utilization
# 使用之前通过keras生成的模型进行数据集的分析，并提取信息

from keras.models import load_model
import numpy as np
import pandas as pd
import jieba
import gensim, logging
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import f1_score
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
from datetime import datetime

# load model
MLPmodel = load_model("outputs/MLPmodel.h5")
LSTMmodel = load_model("outputs/LSTMmodel.h5")


def score(model,text):
    def getWordVecs(wordList):
        W2Vmodel = word2vec.Word2Vec.load("W2Vmodel\CommentsW2V.model")
        vecs = []
        for word in wordList:
            word = word.replace("\n","")
            try:
                vecs.append(W2Vmodel[word])
            except KeyError:
                continue
        return np.array(vecs,dtype="float")
    try:
        seg_list = jieba.cut(text,cut_all=False)
        w2vTest = getWordVecs(list(seg_list))
        shapedVector = sum(np.array(w2cTest))/len(w2cTest)
        seg_list = jieba.cut(text, cut_all=False)
        w2cTest = getWordVecs(list(seg_list))
        shapedVector = sum(np.array(w2cTest))/len(w2cTest)
        return model.predict(np.array([shapedVector]))[0][0]
    except Exception as err:
        seg_list = jieba.cut(text, cut_all=False)
        w2cTest = getWordVecs(list(seg_list))
        shapedVector = sum(np.array(w2cTest))/len(w2cTest)
        shapedVector = np.array([shapedVector])
        reshapedVector = np.reshape(shapedVector,(shapedVector.shape[0],1,shapedVector.shape[1]))
        return model.predict(reshapedVector)[0][0]

def score2(text):
    return (score(MLPmodel,text)+score(LSTMmodel,text))/2


