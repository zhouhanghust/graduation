import jieba.posseg as pseg
import numpy as np
import pandas as pd
import glob
import jieba
import re
import pickle


files = glob.glob("../data_source/*.npy")
data = []
for each in files:
    temp = np.load(each)
    for lieach in temp:
        if not lieach.get("content"):
            continue
        elif lieach.get("content") == "此用户未填写评价内容":
            continue
        else:
            data.append(lieach)

df = pd.DataFrame(data)


positive = sum(df['score']>3) # 9956
negative = len(df) - positive # 337


# 开始清洗数据

MARK_RE = u'[,:;\\\"\\\'\!，。：；“”‘’！]'


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def chinese_word_extraction_POS(content_raw):
    stopwords = stopwordslist('../data_source/ch_stopwords.txt')
    chinese_pattern = u'([\u4e00-\u9fa5]+|' + MARK_RE + '+)'
    chi_pattern = re.compile(chinese_pattern)
    re_data = chi_pattern.findall(content_raw)
    content_clean = ''.join(re_data)
    content_clean = pseg.cut(content_clean) #调用词性标注的方法
    clean_list = []
    for word in content_clean:
        if word.word not in stopwords and word.flag in ('nr','n','v'):
            strs = re.sub(MARK_RE, '', word.word).strip(' ')
            if not strs == '':
                clean_list.append(strs)
    return ' '.join(clean_list)


df['content_pos'] = df[['content']].applymap(chinese_word_extraction_POS)
df['score'] = (df['score'] > 3).map(int)


#------------------------这里用原来的清洗方法得到原来的index----------------------------


def chinese_word_extraction(content_raw):
    stopwords = stopwordslist('../data_source/ch_stopwords.txt')
    chinese_pattern = u'([\u4e00-\u9fa5]+|' + MARK_RE + '+)'
    chi_pattern = re.compile(chinese_pattern)
    re_data = chi_pattern.findall(content_raw)
    content_clean = ''.join(re_data)
    content_clean = jieba.cut(content_clean, cut_all=False)
    clean_list = []
    for word in content_clean:
        if word not in stopwords:
            strs = re.sub(MARK_RE, '', word).strip(' ')
            if not strs == '':
                clean_list.append(strs)
    return ' '.join(clean_list)


df['content_ori'] = df[['content']].applymap(chinese_word_extraction)


def filter_line(line):
    tmp = line.split(" ")
    count = len(tmp)
    if count < 3:
        return False
    return True


index = df['content_ori'].map(filter_line)
df = df[index]
df.drop(['content','content_ori'], axis=1,inplace=True)

with open('../cutted_umbalanced_data/cutted_umbalanced_data_POS.pkl', 'wb') as f:
    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)