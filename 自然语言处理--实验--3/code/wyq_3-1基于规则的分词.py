
# 代码4–1
def RMM(text):
    # 读取词典
    dictionary = []
    dic_path = '../data/dic.utf8'
    for line in open(dic_path, 'r', encoding='utf-8-sig'):
        line = line.strip()
        if not line:
            continue
        dictionary.append(line)
    dictionary = list(set(dictionary))
    # 获取词典最大长度
    max_length = 0
    word_length = []
    for word in dictionary:
        word_length.append(len(word))
    max_length = max(word_length)
    # 切分文本
    cut_list = []
    text_length = len(text)
    while text_length > 0:
        j = 0
        for i in range(max_length, 0, -1):
            if text_length - i < 0:
                continue
            new_word = text[text_length - i:text_length]
            if new_word in dictionary:
                cut_list.append(new_word)
                text_length -= i
                j += 1
                break
        if j == 0:
            text_length -= 1
    cut_list.reverse()
    print(cut_list)
RMM('北京市民办高中')




