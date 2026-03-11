# 代码9-2
import nltk.classify as cf
import nltk.classify.util as cu
import jieba
def setiment(sentences):
    # 文本转换为特征及特征选取
    pos_data = []
    with open('../data/pos.txt', 'r+', encoding='utf-8') as pos:  # 读取积极评论
        while True:
            words = pos.readline()
            if words:
                positive = {}  # 创建积极评论的词典
                words = jieba.cut(words)  # 对评论数据结巴分词
                for word in words:
                    positive[word] = True
                pos_data.append((positive, 'POSITIVE'))  # 对积极词赋予POSITIVE标签
            else:
                break
    neg_data = []
    with open('../data/neg.txt', 'r+', encoding='utf-8') as neg:  # 读取消极评论
        while True:
            words = neg.readline()
            if words:
                negative = {}  # 创建消极评论的词典
                words = jieba.cut(words)  # 对评论数据结巴分词
                for word in words:
                    negative[word] = True
                neg_data.append((negative, 'NEGATIVE'))  # 对消极词赋予NEGATIVE标签
            else:
                break
    # 划分训练集（80%）与测试集（20%）
    pos_num, neg_num = int(len(pos_data) * 0.8), int(len(neg_data) * 0.8)
    train_data = pos_data[: pos_num] + neg_data[: neg_num]  # 抽取80%数据
    test_data = pos_data[pos_num: ] + neg_data[neg_num: ]  # 剩余20%数据
    # 构建分类器（朴素贝叶斯）
    model = cf.NaiveBayesClassifier.train(train_data)
    ac = cu.accuracy(model, test_data)
    print('准确率为：' + str(ac))
    tops = model.most_informative_features()  # 信息量较大的特征
    print('\n信息量较大的前10个特征为:')  
    for top in tops[: 10]:
        print(top[0])  
    for sentence in sentences:
        feature = {}
        words = jieba.cut(sentence)
        for word in words:
            feature[word] = True
        pcls = model.prob_classify(feature)
        sent = pcls.max()  # 情绪面标签（POSITIVE或NEGATIVE）
        prob = pcls.prob(sent)  # 情绪程度
        print('\n','‘',sentence,'’', '的情绪面标签为', sent, '概率为','%.2f%%' % round(prob * 100, 2))
if __name__ == '__main__':
    # 测试
    sentences = ['破烂平板', '手感不错，推荐购买', '刚开始吧还不错，但是后面越来越卡，差评',
                 '哈哈哈哈，我很喜欢', '今天很开心']
    setiment(sentences)



# 代码9-3
from snownlp import SnowNLP  # 调用情感分析函数
# 创建snownlp对象，设置要测试的语句
s1 = SnowNLP('这东西真的挺不错的')
s2 = SnowNLP('垃圾东西')
print('调用sentiments方法获取s1的积极情感概率为:',s1.sentiments)
print('调用sentiments方法获取s2的积极情感概率为:',s2.sentiments)
