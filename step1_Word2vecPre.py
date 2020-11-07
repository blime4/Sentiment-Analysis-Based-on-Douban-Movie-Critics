import os
os.path.dirname(os.__file__)

# 解决step2 Can't get attribute 'Word2VecKeyedVectors' on <module 'gensim.models.keyedvectors' >
from enum import IntEnum
class GensimTrainingAlgo(IntEnum):
    SG = 1
    CBOW = 0


from gensim.models import word2vec
import logging

# 参考链接：
# word2vec的应用----使用gensim来训练模型 https://blog.csdn.net/qq_35273499/article/details/79098689
# 

# 设置日志打印格式
logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s',level=logging.INFO)
# 加载语料库
sentences = word2vec.Text8Corpus("dataset/CommentsCorpus627.txt")
# size 词向量维度 默认100
# 训练模型
# model = word2vec.Word2Vec(sentences,size=200)
model = word2vec.Word2Vec(sentences,size=200,sg=GensimTrainingAlgo.SG)
# 保存模型
model.save("W2Vmodel/CommentsW2V.model")


## 测试一下模型的效果


