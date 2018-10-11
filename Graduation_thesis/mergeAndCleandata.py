# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import glob
import jieba
import re
import pickle

files = glob.glob("./data_source/*.npy")

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
# print(len(df))  # 10293

positive = sum(df['score']>3) # 9956
negative = len(df) - positive # 337


# 开始清洗数据

MARK_RE = u'[,:;\\\"\\\'\!，。：；“”‘’！]'


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def chinese_word_extraction(content_raw):
    stopwords = stopwordslist('./data_source/ch_stopwords.txt')
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


df['content'] = df[['content']].applymap(chinese_word_extraction)
df['score'] = (df['score'] > 3).map(int)


def filter_line(line):
    tmp = line.split(" ")
    count = len(tmp)
    if count < 6:
        return False
    return True


index = df['content'].map(filter_line)
df = df[index]


with open('./cutted_umbalanced_data/cutted_umbalanced_data.pkl', 'wb') as f:
    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)













