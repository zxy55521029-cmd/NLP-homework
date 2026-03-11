# 代码8-9
import re
import os
import json
import jieba
import pandas as pd
from sklearn.cluster import KMeans
import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# 数据读取
files = os.listdir('../data/json/')  # 读取文件列表
train_data = pd.DataFrame()
test_data = pd.DataFrame()
for file in files:
    with open('../data/json/' + file, 'r', encoding='utf-8') as load_f:
        content = []
        while True:
            load_f1 = load_f.readline()
            if load_f1:
                load_dict = json.loads(load_f1)
                content.append(re.sub('[\t\r\n]', '', load_dict['contentClean']))
            else:
                break
        contents = pd.DataFrame(content)
        contents[1] = file[:len(file) - 5]
    # 划分训练集与测试集
    train_data = pd.concat([train_data, contents[:400]], ignore_index=True)
    test_data = pd.concat([test_data, contents[400:]], ignore_index=True)




# 代码8-10
def seg_word(data):
    corpus = []  # 语料库
    stop = pd.read_csv('../data/stopwords.txt', sep='bucunzai', encoding ='utf-8', header=None)
    stopwords = [' '] + list(stop[0])  # 加上空格符号
    for i in range(len(data)):
        string = data.iloc[i, 0].strip()
        seg_list = jieba.cut(string, cut_all=False)  # 结巴分词
        corpu = []
        # 去除停用词
        for word in seg_list:
            if word not in stopwords:
                corpu.append(word)
        corpus.append(' '.join(corpu))
    return corpus
train_corpus = seg_word(train_data)  # 训练语料
test_corpus = seg_word(test_data)  # 测试语料



# 代码8-11
# 将文本中的词语转换为词频矩阵，矩阵元素a[i][j]表示j词在i类文本下的词频
vectorizer = CountVectorizer()
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
train_tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus))
test_tfidf = transformer.fit_transform(vectorizer.fit_transform(test_corpus))
# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
train_weight = train_tfidf.toarray()
test_weight = test_tfidf.toarray()



# 代码8-12
# K-Means聚类
clf = KMeans(n_clusters=4, random_state=4)  # 选择4个中心点
# clf.fit(X)可以将数据输入到分类器里
clf.fit(train_weight)
# 4个中心点
print('4个中心点为:' + str(clf.cluster_centers_))
# 保存模型
joblib.dump(clf, 'km.pkl')
train_res = pd.Series(clf.labels_).value_counts()
s = 0
for i in range(len(train_res)):
    s += abs(train_res[i] - 400)
acc_train = (len(train_res) * 400 - s) / (len(train_res) * 400)
print('\n训练集准确率为：' + str(acc_train))
print('\n每个样本所属的簇为', i + 1, ' ', clf.labels_[i])
for i in range(len(clf.labels_)):
    print(i + 1, ' ', clf.labels_[i])



# 代码8-13
test_res = pd.Series(clf.fit_predict(test_weight)).value_counts()
s = 0
for i in range(len(test_res)):
    s += abs(test_res[i] - 100)
acc_test = (len(test_res) * 100 - s) / (len(test_res) * 100)
print('测试集准确率为：' + str(acc_test))



