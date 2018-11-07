# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.font_manager import FontProperties







with open('./cleaned_data/data_city.pkl','rb') as f:
    data_city = pickle.load(f)



dct = {'beijing':'北京','shanghai':'上海','hangzhou':'杭州','guangzhou':'广州','xiamen':'厦门','chengdu':'成都'}


font = FontProperties(fname="simhei.ttf",size=18)
fig,axes = plt.subplots(3,2)
for i,city in enumerate(data_city.keys()):
    temp = data_city[city][1]
    temp.sort()
    axes[i//2,i%2].scatter(np.arange(len(temp)), temp)
    xlim = int(axes[i//2,i%2].get_xlim()[1])
    axes[i // 2, i % 2].set_xlim(0)
    axes[i // 2, i % 2].set_xticks([0,xlim//5,xlim*2//5,xlim*3//5,xlim*4//5,xlim])
    axes[i // 2, i % 2].set_title('%s的情感得分'%dct[city],fontproperties=font)
    axes[i // 2, i % 2].grid()



    
fig.set_size_inches(10.5, 10.5) 
fig.tight_layout()
# ax = plt.gca()
# ax.patch.set_facecolor("blue")
# ax.patch.set_alpha(0.5)

# plt.show()


fig.savefig('./pic_test.png',dpi=250)















