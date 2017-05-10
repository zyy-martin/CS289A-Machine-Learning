from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


train_X = np.loadtxt('mnist_train_data.txt', dtype=int)
train_Y = np.loadtxt('mnist_train_label.txt', dtype=int)
valid_X = np.loadtxt('mnist_valid_data.txt', dtype=int)
valid_Y = np.loadtxt('mnist_valid_label.txt', dtype=int)
err_valid = []
num = []
print('start...')
for i in range(10):
    c = (i+5)*10 ** (-7)
    clf = SVC(C=c, kernel='linear')
    clf.fit(train_X[:10000, :], train_Y[:10000])
    predict_Y = clf.predict(valid_X)
    err_valid.append(1 - accuracy_score(valid_Y, predict_Y))
    num.append(c)
    print(err_valid[i])

plt.plot(num, err_valid, 'rs', label='validation error' )
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend()
plt.xlabel('c value')
plt.ylabel('error')
plt.title('error vs c value')
plt.savefig('p3_3.png')
