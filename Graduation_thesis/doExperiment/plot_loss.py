# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pickle


with open('./loss.pkl','rb') as f:
    loss = pickle.load(f)

plt.figure()

font = FontProperties(fname="simhei.ttf",size=18)
plt.plot(loss, c='b')
plt.xlabel('训练次数', fontproperties=font)
plt.ylabel('损失函数值', fontproperties=font)


plt.tight_layout()
# plt.show()
plt.savefig('./pic_loss.png',dpi=250)

















