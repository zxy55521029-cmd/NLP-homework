
#代码4–2
import jieba
# text = '中文分词是自然语言处理的一部分！'
text = "我来到北京清华大学"
seg_list = jieba.cut(text, cut_all=True)
print('全模式：', '/ ' .join(seg_list))
seg_list = jieba.cut(text, cut_all=False)
print('精确模式：', '/ '.join(seg_list))
seg_list = jieba.cut_for_search(text)
print('搜索引擎模式', '/ '.join(seg_list))
