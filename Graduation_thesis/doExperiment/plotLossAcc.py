import pickle
import matplotlib.pyplot as plt


with open("../BuildModel/LossAcc/loss.pkl","rb") as f:
    loss_lst = pickle.load(f)

with open("../BuildModel/LossAcc/acc.pkl","rb") as f:
    acc_lst = pickle.load(f)

'''
plt.figure()
plt.plot(loss_lst[0], 'g-', label='loss-train')
plt.plot(loss_lst[1], 'r--', label='loss-test')

plt.xlim(0, max(len(loss_lst[0]), len(loss_lst[1])))
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(fontsize=10)
plt.show()
'''

plt.figure()
plt.plot(acc_lst[0], 'g-', label='loss-train')
plt.plot(acc_lst[1], 'r--', label='loss-test')

plt.xlim(0, max(len(acc_lst[0]), len(acc_lst[1])))
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.legend(fontsize=10)
plt.show()



