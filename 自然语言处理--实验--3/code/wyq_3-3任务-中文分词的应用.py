# 代码4–3
import os
# 若要二次运行，则需删除已生成的json文件，否则会继续对原文件写入内容并出现解析错误
import json
import datetime

text = '学校是学习的好地方！'


def train():
    # 初始化参数
    model = '../tmp/hmm_model.json'
    # 先检查当前路径下是否有json文件，如果有json文件，需要删除
    if os.path.exists(model):
        f = open(model, 'r')
        data=json.loads(f.read())
        trans_prob = data[0]
        emit_prob = data[1]
        init_prob = data[2]
        f.close()
    else:
        trans_prob = {}  # 转移概率
        emit_prob = {}  # 发射概率
        init_prob = {}  # 状态出现次数
    Count_dict = {}
    state_list = ['B', 'M', 'E', 'S']
    for state in state_list:
        trans = {}
        for s in state_list:
            trans[s] = 0
        trans_prob[state] = trans
        emit_prob[state] = {}
        init_prob[state] = 0
        Count_dict[state] = 0
    count = -1
    # 读取并处理单词、计算概率矩阵
    path = '../data/trainCorpus.txt'
    for line in open(path, 'r'):
        count += 1
        line = line.strip()
        if not line:
            continue

        # 读取每一行的单词
        word_list = []
        for i in line:
            if i != ' ':
                word_list.append(i)

        # 标注每个单词的位置标签
        word_label = []
        for word in line.split():
            label = []
            if len(word) == 1:
                label.append('S')
            else:
                label += ['B'] + ['M'] * (len(word) - 2) + ['E']
            word_label.extend(label)

        # 统计各个位置状态下的出现次数，用于计算概率
        for index, value in enumerate(word_label):
            Count_dict[value] += 1
            if index == 0:
                init_prob[value] += 1
            else:
                trans_prob[word_label[index - 1]][value] += 1
                emit_prob[word_label[index]][word_list[index]] = (
                        emit_prob[word_label[index]].get(
                            word_list[index], 0) + 1.0)
    # 初始概率
    for key, value in init_prob.items():
        init_prob[key] = value * 1 / count
        # 转移概率
    for key, value in trans_prob.items():
        for k, v in value.items():
            value[k] = v / Count_dict[key]
        trans_prob[key] = value
    # 发射概率，采用加1平滑
    for key, value in emit_prob.items():
        for k, v in value.items():
            value[k] = (v + 1) / Count_dict[key]
        emit_prob[key] = value
    # 将3个概率矩阵保存至json文件
    model = '../tmp/hmm_model.json'
    param_list = [trans_prob, emit_prob, init_prob]
    f = open(model, 'w', encoding="utf-8")
    # f.write(json.dumps(trans_prob) + '\n' + json.dumps(emit_prob) +
    #         '\n' + json.dumps(init_prob))
    json.dump(param_list, f, indent=4, ensure_ascii=False)
    f.close()


# 代码4–4
def viterbi(text, state_list, init_prob, trans_prob, emit_prob):
    V = [{}]
    path = {}
    # 初始概率
    for state in state_list:
        V[0][state] = init_prob[state] * emit_prob[state].get(text[0], 0)
        path[state] = [state]

    # 当前语料中所有的字
    key_list = []
    for key, value in emit_prob.items():
        for k, v in value.items():
            key_list.append(k)

    # 计算待分词文本的状态概率值，得到最大概率序列
    for t in range(1, len(text)):
        V.append({})
        newpath = {}
        for state in state_list:
            if text[t] in key_list:
                emit_count = emit_prob[state].get(text[t], 0)
            else:
                emit_count = 1
            (prob, a) = max(
                [(V[t - 1][s] * trans_prob[s].get(state, 0) * emit_count, s)
                 for s in state_list if V[t - 1][s] > 0])
            V[t][state] = prob
            newpath[state] = path[a] + [state]
        path = newpath
    # 根据末尾字的状态，判断最大概率状态序列
    if emit_prob['M'].get(text[-1], 0) > emit_prob['S'].get(text[-1], 0):
        (prob, a) = max([(V[len(text) - 1][s], s) for s in ('E', 'M')])
    else:
        (prob, a) = max([(V[len(text) - 1][s], s) for s in state_list])

    return (prob, path[a])


# 代码4–5
def cut(text):
    state_list = ['B', 'M', 'E', 'S']
    model = '../tmp/hmm_model.json'
    # 先检查当前路径下是否有json文件，如果有json文件，需要删除
    if os.path.exists(model):
        f = open(model, 'r', encoding="utf-8")
        data=json.loads(f.read())
        trans_prob = data[0]
        emit_prob = data[1]
        init_prob = data[2]
        f.close()
    else:
        trans_prob = {}
        emit_prob = {}
        init_prob = {}
    # 利用维特比算法，求解最大概率状态序列
    prob, pos_list = viterbi(text, state_list, init_prob, trans_prob, emit_prob)
    # 判断待分词文本每个字的状态，输出结果
    begin, follow = 0, 0
    for index, char in enumerate(text):
        state = pos_list[index]
        if state == 'B':
            begin = index
        elif state == 'E':
            yield text[begin: index + 1]
            follow = index + 1
        elif state == 'S':
            yield char
            follow = index + 1
    if follow < len(text):
        yield text[follow:]


# 训练、分词
starttime = datetime.datetime.now()
train()
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)

cut(text)
print(text)
print(str(list(cut(text))))

# 代码 4–6
import jieba


def word_extract():
    # 读取文件
    corpus = []
    path = '../data/news.txt'
    content = ''
    for line in open(path, 'r', encoding='utf-8', errors='ignore'):
        line = line.strip()
        content += line
    corpus.append(content)
    # 加载停用词
    stop_words = []
    path = '../data/stopword.txt'
    for line in open(path, encoding='utf8'):
        line = line.strip()
        stop_words.append(line)
        # jieba分词
    split_words = []
    word_list = jieba.cut(corpus[0])
    for word in word_list:
        if word not in stop_words:
            split_words.append(word)
    # 提取前10个高频词
    dic = {}
    word_num = 10
    for word in split_words:
        dic[word] = dic.get(word, 0) + 1
    freq_word = sorted(dic.items(), key=lambda x: x[1],
                       reverse=True)[: word_num]
    print('样本：' + corpus[0])
    print('样本分词效果：' + '/ '.join(split_words))
    print('样本前10个高频词：' + str(freq_word))


word_extract()
