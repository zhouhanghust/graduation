from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle

with open("../BuildModel_CRNN_Attention/predAndy_test.pkl", "rb") as f:
    data = pickle.load(f)

y_pred = data[0]
y_test = data[1]


def my_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
    total = 0
    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]:
            total += 1
    print("Accuracy: %s"%(total/len(y_pred)))
    print("第一行是y_true为0，第二行是y_true为1，第一列是y_pred为0，第二列是y_pred为1")
    print(conf_mat)


my_confusion_matrix(y_test,y_pred)


print("classification_report(left: labels):")
print(classification_report(y_test, y_pred))
print("第二行 标签为1 的是需要的数据")
# presicion=TP/(TP+FP) recall = TP/(TP+FN)
'''
FN：False Negative,被判定为负样本，但事实上是正样本。
FP：False Positive,被判定为正样本，但事实上是负样本。
TN：True Negative,被判定为负样本，事实上也是负样本。
TP：True Positive,被判定为正样本，事实上也是证样本。
'''

