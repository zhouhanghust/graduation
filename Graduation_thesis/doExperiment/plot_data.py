# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import glob
import pandas as pd


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
score = df.score.tolist()

# font = FontProperties(fname="simhei.ttf",size=18)
font = FontProperties(size=18)
score.sort()

ax = plt.subplot(111)
ax.scatter(np.arange(len(score)), score)
xlim = int(ax.get_xlim()[1])
ax.set_xlim(0)
# ax.set_ylim(0)
ax.set_xticks([0,xlim//5,xlim*2//5,xlim*3//5,xlim*4//5,xlim])
ax.set_title('Sentiment Score', fontproperties=font)
ax.grid()


plt.savefig('./SentimentScore.png',dpi=250)

