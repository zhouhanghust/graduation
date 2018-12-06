import pickle
import matplotlib.pyplot as plt
import numpy as np


with open("../cuted_unbalanced_data/label.pkl",'rb') as f:
    label = pickle.load(f)

total_1 = sum(label)
total_0 = len(label) - total_1
result = [total_0,total_1]
index = [0,1]
rects = plt.bar(left=index,height=result,width=0.8)
# plt.ylabel('NumOf',size=14)
# plt.xlabel('TopicIndex',size=14)
plt.xticks(index,size=10)
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom", size=6)
plt.savefig('./unbalanced_label_distributed.png',dpi=250)
