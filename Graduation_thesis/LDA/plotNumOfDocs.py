import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("../cutted_umbalanced_data/topicLst.pkl","rb") as f:
    topicLst = pickle.load(f)

dct = {}
for each in topicLst:
    if each in dct:
        dct[each] += 1
    else :
        dct[each] = 1

dct = sorted(dct.items(),key=lambda item:item[0])
result = [each[1] for each in dct]
index = np.arange(1,len(result)+1)
rects = plt.bar(left=index,height=result,width=0.9)
plt.ylabel('NumOfDocs',size=14)
plt.xlabel('TopicIndex',size=14)
plt.xticks(np.arange(1,len(result)+1),size=6)
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom", size=6)
plt.savefig('./pic_idf.png',dpi=250)
