import pickle
import matplotlib.pyplot as plt


with open("../BuildModel/LossAcc/loss.pkl","rb") as f:
    loss_lst_cnn = pickle.load(f)

with open("../BuildModel_RNN/LossAcc/loss.pkl","rb") as f:
    loss_lst_rnn = pickle.load(f)

with open("../BuildModel_CRNN_Attention/LossAcc/loss.pkl","rb") as f:
    loss_lst_crnn = pickle.load(f)


plt.figure(figsize=(12,4))
plt.subplots_adjust(left=0.05,right=0.95,wspace=0.15)
plt.subplot(121)
plt.plot(loss_lst_cnn[0], 'r-', label='textcnn')
plt.plot(loss_lst_rnn[0], 'g-', label='rnn_attention')
plt.plot(loss_lst_crnn[0], 'b-', label='crnn_attention')

plt.xlim(0, max(len(loss_lst_cnn[0]), len(loss_lst_rnn[0]),len(loss_lst_crnn[0])))
plt.xlabel('iteration')
plt.ylabel('loss-train')
plt.legend(fontsize=10)


plt.subplot(122)
plt.plot(loss_lst_cnn[1], 'r-', label='textcnn')
plt.plot(loss_lst_rnn[1], 'g-', label='rnn_attention')
plt.plot(loss_lst_crnn[1], 'b-', label='crnn_attention')

plt.xlim(0, max(len(loss_lst_cnn[1]), len(loss_lst_rnn[1]),len(loss_lst_crnn[1])))
plt.xlabel('iteration')
plt.ylabel('loss-test')
plt.legend(fontsize=10)

plt.savefig('./loss_comparison.png',dpi=250)
