# -*- coding: utf-8 -*-

import pickle
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


with open('../tokenizer_smote/tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)

word_counts = tokenizer.word_counts
word_counts = dict(word_counts)

countlist = sorted(word_counts.items(),key=lambda item:item[1])
data_pre = countlist[-15:]
print(data_pre)

fig = plt.figure(1)
font = FontProperties(fname="simhei.ttf", size=16)
ax1 = plt.subplot(111)

data = [each[1] for each in data_pre]
xlabels = [each[0] for each in data_pre]
width = 0.8
x_bar = np.arange(len(data))

rect=ax1.bar(left=x_bar,height=data,width=width,color="lightblue")

for rec in rect:
    x=rec.get_x()
    height=int(rec.get_height())
    ax1.text(x+0.1,1.02*height,str(height))

ax1.set_xticks(x_bar)
ax1.set_xticklabels(xlabels,fontproperties=font)
ax1.set_ylabel("frequency")
# ax1.set_title("The Sales in 2016")
# ax1.grid(True)
# ax1.set_ylim(0,28)

plt.show()
