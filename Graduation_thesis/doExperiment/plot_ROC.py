import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc


with open("../BuildModel/predict_prob_label.pkl","rb") as f:
    predict_prob_label_cnn = pickle.load(f)

with open("../BuildModel_RNN/predict_prob_label.pkl","rb") as f:
    predict_prob_label_rnn = pickle.load(f)

with open("../BuildModel_CRNN_Attention/predict_prob_label.pkl","rb") as f:
    predict_prob_label_crnn = pickle.load(f)


predict_prob_cnn, labels_cnn = predict_prob_label_cnn
predict_prob_rnn, labels_rnn = predict_prob_label_rnn
predict_prob_crnn, labels_crnn = predict_prob_label_crnn

false_positive_rate_cnn,true_positive_rate_cnn,thresholds_cnn=roc_curve(labels_cnn, predict_prob_cnn)
roc_auc_cnn=auc(false_positive_rate_cnn, true_positive_rate_cnn)

false_positive_rate_rnn,true_positive_rate_rnn,thresholds_rnn=roc_curve(labels_rnn, predict_prob_rnn)
roc_auc_rnn=auc(false_positive_rate_rnn, true_positive_rate_rnn)

false_positive_rate_crnn,true_positive_rate_crnn,thresholds_crnn=roc_curve(labels_crnn, predict_prob_crnn)
roc_auc_crnn=auc(false_positive_rate_crnn, true_positive_rate_crnn)

plt.plot(false_positive_rate_crnn, true_positive_rate_crnn,'r-',label='AUC-cnn = %0.4f'% roc_auc_crnn)
plt.plot(false_positive_rate_rnn, true_positive_rate_rnn,'g-',label='AUC-rnn = %0.4f'% roc_auc_rnn)
plt.plot(false_positive_rate_cnn, true_positive_rate_cnn,'b-',label='AUC-crnn = %0.4f'% roc_auc_cnn)

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.savefig('./ROC.png',dpi=250)








