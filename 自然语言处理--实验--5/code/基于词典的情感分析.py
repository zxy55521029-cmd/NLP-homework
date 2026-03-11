# 代码9-1
import re
import jieba
import codecs
from collections import defaultdict  # 导入collections用于创建空白词典

def seg_word(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = []
    for word in seg_list:
        seg_result.append(word)
        stopwords = set()
        stopword = codecs.open('../data/stopwords.txt', 'r',
                               encoding='utf-8')  # 加载停用词
    for word in stopword:
        stopwords.add(word.strip())
    stopword.close()
    return list(filter(lambda x: x not in stopwords, seg_result))

def sort_word(word_dict):
    sen_file = open('../data/BosonNLP_sentiment_score.txt', 'r+',
                    encoding='utf-8')  # 加载Boson情感词典
    sen_list = sen_file.readlines()
    sen_dict = defaultdict()  # 创建词典
    for s in sen_list:
        s = re.sub('\n', '', s)  # 去除每行最后的换行符
        if s:
            # 构建以key为情感词，value为对应分值的词典
            sen_dict[s.split(' ')[0]] = s.split(' ')[1]
    not_file = open('../data/否定词.txt', 'r+',
                    encoding='utf-8')  # 加载否定词词典
    not_list = not_file.readlines()
    for i in range(len(not_list)):
        not_list[i] = re.sub('\n', '', not_list[i])
    degree_file = open('../data/程度副词（中文）.txt', 'r+',
                       encoding='utf-8')  # 加载程度副词词典
    degree_list = degree_file.readlines()
    degree_dic = defaultdict()
    for d in degree_list:
        d = re.sub('\n', '', d)
        if d:
            degree_dic[d.split(' ')[0]] = d.split(' ')[1]
    sen_file.close()
    degree_file.close()
    not_file.close()
    sen_word = dict()
    not_word = dict()
    degree_word = dict()
    # 分类
    for word in word_dict.keys():
        if word in sen_dict.keys() and word not in not_list and word not in degree_dic.keys():
            sen_word[word_dict[word]] = sen_dict[word]  # 情感词典中的包含分词结果的词
        elif word in not_list and word not in degree_dic.keys():
            not_word[word_dict[word]] = -1  # 程度副词词典中的包含分词结果的词
        elif word in degree_dic.keys():
            # 否定词典中的包含分词结果的词
            degree_word[word_dict[word]] = degree_dic[word]
    return sen_word, not_word, degree_word  # 返回分类结果

def list_to_dict(word_list):
    data = {}
    for x in range(0, len(word_list)):
        data[word_list[x]] = x
    return data

def socre_sentiment(sen_word, not_word, degree_word, seg_result):
    W = 1  # 初始化权重
    score = 0
    sentiment_index = -1  # 情感词下标初始化
    for i in range(0, len(seg_result)):
        if i in sen_word.keys():
            score += W * float(sen_word[i])
            sentiment_index += 1  # 下一个情感词
            for j in range(len(seg_result)):
                if j in not_word.keys():
                    score *= -1  # 否定词反转情感
                elif j in degree_word.keys():
                    score *= float(degree_word[j])  # 乘以程度副词
    return score

def setiment(sentence):
    # 对文本进行分词和去停用词，去除跟情感词无关的词语
    seg_list = seg_word(sentence)
    # 对分词结果进行分类，找出其中的情感词、程度副词和否定词
    sen_word, not_word, degree_word = sort_word(list_to_dict(seg_list))
    # 计算并汇总情感词的得分
    score = socre_sentiment(sen_word, not_word, degree_word, seg_list)
    return seg_list, sen_word, not_word, degree_word, score

if __name__ == '__main__':
    print(setiment('我今天特别开心'))
    print(setiment('我今天很开心、非常兴奋'))
    print(setiment('我昨天开心，今天不开心'))