from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


data = sio.loadmat('hw01_data/spam/spam_data.mat')
train_data_spam = np.array(data['training_data'])
train_label_spam = np.array(data['training_labels'])
data_spam = np.concatenate((train_data_spam, train_label_spam.T), axis=1)
np.random.shuffle(data_spam)

train_data_spam = data_spam[:,:-1]
train_label_spam = data_spam[:,-1]

num = []
scores = []
for i in range(20):
    c = (i+1)*0.5
    c = num[i]
    clf = SVC(C=c, kernel='linear')
    score = cross_val_score(clf, train_data_spam, train_label_spam, cv=5, scoring='accuracy')
    scores.append(1-np.mean(score))
    print(i)
    num.append(c)
print(scores)
plt.plot(num, scores, 'rs', label='mean error' )
plt.legend()
plt.xlabel('c value')
plt.ylabel('error')
plt.title('error vs c value')
plt.savefig('p4_1.png')
plt.show()
