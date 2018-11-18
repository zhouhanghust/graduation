import pickle
import numpy as np
import random

with open('./cutted_umbalanced_data_POS.pkl', 'rb') as f:
    data_pos = pickle.load(f)

with open('../cutted_umbalanced_data/cutted_umbalanced_data.pkl', 'rb') as f:
    data = pickle.load(f)

data_pos = data_pos.content_pos.values
data_normal = data.content.values

label = data.score.values

neg = np.asarray(label,dtype=np.int32) == 0
negInd = (np.where(neg))[0]
posInd = (np.where(np.asarray(label,dtype=np.int32)))[0]

ratio = len(posInd) // len(negInd)
negInd = np.tile(negInd,ratio)


inds = np.concatenate([posInd, negInd], axis=0)


newT = []
newT_POS = []
newL = []
for each in inds:
    newT.append(data_normal[each])
    newT_POS.append(data_pos[each])
    newL.append(label[each])
print(len(newL))
print(sum(newL)/len(newL))

with open("./balanced_data.pkl","wb") as handle:
    pickle.dump(newT, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("./balanced_data_pos.pkl","wb") as handle:
    pickle.dump(newT_POS, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("./balanced_data_label.pkl","wb") as handle:
    pickle.dump(newL, handle, protocol=pickle.HIGHEST_PROTOCOL)