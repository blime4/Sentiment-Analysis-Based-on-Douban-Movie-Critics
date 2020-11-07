# 参考链接
# 主成分分析法 http://c.biancheng.net/view/1951.html
# 关于sklearn.preprocessing中scale和StandardScaler的使用 https://blog.csdn.net/dengdengma520/article/details/79629186
# 机器学习—保存模型、加载模型—Joblib https://blog.csdn.net/weixin_45252110/article/details/98883571
# Scipy教程 - 统计函数库scipy.stats https://blog.csdn.net/pipisorry/article/details/49515215
# word2vec的应用----使用gensim来训练模型 https://blog.csdn.net/qq_35273499/article/details/79098689
# scipy.stats的用法——常见的分布和函数 https://blog.csdn.net/baby_superman/article/details/83749803



import numpy as np
import sys
import re
import codecs # 解码库
import os
import jieba
import gensim,logging
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib # 保存模型、加载模型
import joblib
from sklearn.preprocessing import scale # 直接将给定数据进行标准化。=> 均值为零，标准差为1
from sklearn.svm import SVC # Support Vector Classification 就是支持向量机用于分类
from sklearn.decomposition import PCA # Principal Component Analysis 主成分分析
from sklearn.metrics import f1_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as auc
from scipy import stats # 统计函数库
from scipy.stats import ks_2samp
from keras.models import Sequential 
from keras.layers import Dense,Dropout, Activation 
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
import sklearn

# 加载W2V模型
W2Vmodel = word2vec.Word2Vec.load("W2Vmodel\CommentsW2V.model")

def getWordVecs(wordList):
    vecs = []
    for word in wordList:
        word = word.replace("\n","")
        try:
            vecs.append(W2Vmodel[word])
        except KeyError:
            continue
    return np.array(vecs,dtype="float")

def buildWordVec(filename):
    posInput = []
    with open(filename,"r",encoding="utf8") as textfile:
        for line in textfile:
            if not line.strip():
                continue
            line = line.split(" ")[2:]
            line = line[:-1]
            resultList = getWordVecs(line)
            if len(resultList) > 0:
                resultArray = sum(np.array(resultList))/len(resultList)
                posInput.append(resultArray)
    return posInput

posInput = buildWordVec("dataset\TrainPos.txt")
negInput = buildWordVec("dataset\TrainNeg.txt")


y = np.concatenate((np.ones(len(posInput)),np.zeros(len(negInput))))
X = posInput + negInput
#标准化
X = scale(X) 
# PCA降维成100维
X_reduced = PCA(n_components=100).fit_transform(X)

# 训练集和测试集
X_reduced_train,X_reduced_test,y_reduced_train,y_reduced_test = train_test_split(X_reduced,y,test_size=0.4,random_state=1)
X_trian,X_test,y_trian,y_test = train_test_split(X,y,test_size=0.4,random_state=1)

# 有两个PCA类的成员值得关注。
# 第一个是explained_variance_，它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
# 第二个是explained_variance_ratio_，它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。

# plt.axes 四个参数的话，
# 前两个指的是相对于坐标原点的位置，
# 后两个指的是坐标轴的长/宽度

# 画图分析

pca=PCA(n_components=100)  
pca.fit(X) 
plt.figure(1, figsize=(4, 3)) 
plt.clf() 
plt.axes([.2, .2, .7, .7]) 
plt.plot(pca.explained_variance_, linewidth=2) 
plt.axis('tight') 
plt.xlabel('n_components') 
plt.ylabel('explained_variance_') 

## 通过图，可以很直观的看出，当维度是100的时候
## 每一个维度代表的重要程度都差不多
## 说明每一个维度都是不可或缺的，达到了数据降维的目的。


# pca=PCA(n_components=100)  
# pca.fit(X) 
# plt.figure(1, figsize=(4, 3)) 
# plt.clf() 
# plt.axes([.2, .2, .7, .7]) 
# plt.plot(pca.explained_variance_ratio_, linewidth=2) 
# plt.axis('tight') 
# plt.xlabel('n_components') 
# plt.ylabel('explained_variance_ratio_') 

# 所以使用 n_components = 100
X_reduced = PCA(n_components = 100).fit_transform(X)



# 高斯核和多项式核干的事情截然不同的，如果对于样本数量少，特征多的数据集，高斯核相当于对样本降维；
# 高斯核的任务：找到更有利分类任务的新的空间。
# 高斯核本质是在衡量样本和样本之间的“相似度”，在一个刻画“相似度”的空间中，
# 让同类样本更好的聚在一起，进而线性可分。

# 2.3.1 SVM (RBF) + PCA
# SVM (RBF)分类表现更为宽松，且使用PCA降维后的模型表现有明显提升，
# misclassified多为负向文本被分类为正向文本，其中AUC = 0.92，KSValue = 0.7。

# C越大，对误分类惩罚越大，训练集准确率高，泛化能力弱
# C越小，反之

clf = SVC(C = 2, probability = True) 
clf.fit(X_reduced_train, y_reduced_train) 

print ('Test Accuracy: %.2f'% clf.score(X_reduced_test, y_reduced_test) )

pred_probas = clf.predict_proba(X_reduced_test)[:,1] 
print ("KS value: %f" % ks_2samp(y_reduced_test, pred_probas)[0] )

fpr,tpr,_ = roc_curve(y_reduced_test, pred_probas)
roc_auc =  sklearn.metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'roc_auc = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.show()

joblib.dump(clf, "SVC.pkl")

model = Sequential()
model.add(
    Dense(512,input_dim=200,init="uniform",activation='tanh'),
    Dropout(0.5),
    Dense(256,activation='relu'),
    Dropout(0.5),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(32,activation='relu'),
    Dropout(0.5),
    Dense(1,activation="sigmoid")
)
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
