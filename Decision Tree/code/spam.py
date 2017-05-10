import scipy.io
import numpy as np
import decisionTree
import randomForest
import random
import csv



data = scipy.io.loadmat('dist/spam_data.mat')
test_X = data['test_data']
train_y = (data['training_labels'].T)[:,0]
train_X = data['training_data']
x_y = list(zip(train_X, train_y))
random.shuffle(x_y)
train_X = np.array([e[0] for e in x_y])
train_y = np.ravel([e[1] for e in x_y])
validation_X = train_X[:2000, :]
validation_y = train_y[:2000]
train_X = train_X[2000:, :]
train_y = train_y[2000:]
print(train_X.shape)

# random forest

rf = randomForest.RandomForest(10,10,train_X.shape[0],train_X.shape[1])
rf.train(train_X,train_y)
res = rf.predict(validation_X)

score = 0
for i in range(len(res)):
    if res[i] == validation_y[i]:
        score += 1
score /= len(res)
print(score)

# decision tree

tree = decisionTree.DecisionTree(10,train_X.shape[1])
tree.train(train_X,train_y)
res = tree.predict(validation_X)

score = 0
for i in range(len(res)):
    if res[i] == validation_y[i]:
        score += 1
score /= len(res)
print(score)




# with open('spam_prediction.csv','wt') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerow(['Id','Category'])
#     for i, cat in enumerate(res):
#         writer.writerow([str(i),str(cat)])