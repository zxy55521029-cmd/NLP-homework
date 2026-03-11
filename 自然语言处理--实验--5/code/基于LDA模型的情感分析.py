# 代码9-4
import pandas as pd
from snownlp import SnowNLP
data = pd.read_csv('../data/comment.csv', sep=',', encoding='utf-8', header=0)
comment_data = data.loc[: , ['评论']]  # 只提取评论数据
# 去除重复值
comment_data = comment_data.drop_duplicates()
# 短句删除
comments_data = comment_data.iloc[: , 0]
comments = comments_data[comments_data.apply(len) >= 4]  # 剔除字数少于4的数据
# 语料压缩，句子中常出现重复语句，需要进行压缩
def yasuo(string):
    for i in [1, 2]:
        j = 0
        while j < len(string) - 2 * i:
            if string[j: j + i] == string[j + i: j + 2 * i] and (
                    string[j + i: j + 2 * i] == string[j + i: j + 3 * i]):
                k = j + 2 * i
                while k + i < len(string) and string[j: j + i] == string[j: j + 2 * i]:
                    k += i
                string = string[: j + i] + string[k + i:]
            j += 1
    for i in [3, 4, 5]:
        j = 0
        while j < len(string) - 2 * i:
            if string[j: j + i] == string[j + i: j + 2 * i]:
                k = j + 2 * i
                while k + i < len(string) and string[j: j + i] == string[j: j + 2 * i]:
                    k += i
                string = string[: j + i] + string[k + i:]
            j += 1
    if string[: int(len(string) / 2)] == string[int(len(string) / 2):]:
        string = string[: int(len(string) / 2)]
    return string
comments = comments.astype('str').apply(lambda x: yasuo(x))



# 代码9-5
from gensim import corpora, models, similarities
import jieba
# 情感分析
coms = []
coms = comments.apply(lambda x: SnowNLP(x).sentiments)
# 情感分析，coms在0~1之间，以0.5分界，大于0.5，则为正面情感
pos_data = comments[coms >= 0.6]  # 正面情感数据集，取0.6是为了增强情感
neg_data = comments[coms < 0.4]  # 负面情感数据集
# 分词
mycut = lambda x: ' '.join(jieba.cut(x))  # 自定义简单分词函数
pos_data = pos_data.apply(mycut)
neg_data = neg_data.apply(mycut)
pos_data.head(5)
neg_data.tail(5)
print(f'len(pos_data):{len(pos_data)}')
print(f'len(neg_data):{len(neg_data)}')
# 去停用词
stop = pd.read_csv('../data/stopwords.txt', sep='bucunzai', encoding='utf-8', header=None)
stop = ['', ''] + list(stop[0])  # 添加空格符号，pandas过滤了空格符
pos = pd.DataFrame(pos_data)
neg = pd.DataFrame(neg_data)
pos[1] = pos['评论'].apply(lambda s: s.split(' '))  # 空格分词
pos[2] = pos[1].apply(lambda x: [i for i in x if i not in stop])  # 去除停用词
neg[1] = neg['评论'].apply(lambda s: s.split(' '))
neg[2] = neg[1].apply(lambda x: [i for i in x if i not in stop])
# 正面主题分析
pos_dict = corpora.Dictionary(pos[2])  # 建立词典
pos_corpus = [pos_dict.doc2bow(i) for i in pos[2]]  # 建立语料库
pos_lda = models.LdaModel(pos_corpus, num_topics=3, id2word=pos_dict)  # LDA模型训练
for i in range(3):
    print('pos_topic' + str(i))
    print(pos_lda.print_topic(i))  # 输出每个主题
# 负面主题分析
neg_dict = corpora.Dictionary(neg[2])  # 建立词典
neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]]  # 建立语料库，bag of word
neg_lda = models.LdaModel(neg_corpus, num_topics=3, id2word=neg_dict)  # LDA模型训练
for i in range(3):
    print('neg_topic' + str(i))
    print(neg_lda.print_topic(i))  # 输出每个主题