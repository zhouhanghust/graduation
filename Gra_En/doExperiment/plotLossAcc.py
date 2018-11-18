import pickle
import matplotlib.pyplot as plt


with open("../BuildModel_CRNN_Attention/LossAcc/loss.pkl","rb") as f:
    loss_lst = pickle.load(f)

with open("../BuildModel_CRNN_Attention/LossAcc/acc.pkl","rb") as f:
    acc_lst = pickle.load(f)


plt.figure(0)
plt.plot(loss_lst[0], 'g-', label='loss-train')
plt.plot(loss_lst[1], 'r--', label='loss-test')

plt.xlim(0, max(len(loss_lst[0]), len(loss_lst[1])))
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(fontsize=10)
plt.savefig('../BuildModel_CRNN_Attention/loss.png',dpi=250)


plt.figure(1)
plt.plot(acc_lst[0], 'g-', label='acc-train')
plt.plot(acc_lst[1], 'r--', label='acc-test')

plt.xlim(0, max(len(acc_lst[0]), len(acc_lst[1])))
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.legend(fontsize=10)
plt.savefig('../BuildModel_CRNN_Attention/acc.png',dpi=250)


# plt.tight_layout()
# plt.savefig('./pic_loss.png',dpi=250)
