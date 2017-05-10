import numpy as np
import csv
import pandas as pd
import collections
from sklearn.feature_extraction import DictVectorizer
from decisionTree import DecisionTree
from randomForest import RandomForest
import matplotlib.pyplot as plt


# preprocessing training and testing data
data_test = []
data = []
dict ={}
with open('hw5_census_dist/train_data.csv') as f:
    reader = csv.reader(f,delimiter=',')
    for i, row in enumerate(reader):
        if i == 0:
            for j in range(len(row)):
                dict[j] = row[j]
        else:
            data.append(row)
data = np.array(data)
most_common_dict = {}
for i in range(data.shape[1]):
    most_common_dict[i] = collections.Counter(data[:,i]).most_common(1)[0][0]
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i,j] == '?':
            data[i, j] = most_common_dict[j]

with open('hw5_census_dist/test_data.csv') as f:
    reader = csv.reader(f,delimiter=',')
    for i, row in enumerate(reader):
        if i == 0:
            continue
        else:
            data_test.append(row)
data_test = np.array(data_test)
most_common_dict = {}
for i in range(data_test.shape[1]):
    most_common_dict[i] = collections.Counter(data_test[:,i]).most_common(1)[0][0]
for i in range(data_test.shape[0]):
    for j in range(data_test.shape[1]):
        if data_test[i,j] == '?':
            data_test[i, j] = most_common_dict[j]

print(dict)
non_cat = [0,2,4,10,11,12]
cat = [1,3,5,6,7,8,9,13]
cat_dict = []
cat_data = []
non_cat_data = data[:,non_cat]
non_cat_data = np.array(non_cat_data,dtype='int')
non_cat_data_test = data_test[:,non_cat]
non_cat_data_test = np.array(non_cat_data_test,dtype='int')
for count, i in enumerate(cat):
    cat_dict.append([])
    feature = data[:, i]
    dist_feature = set(feature)
    for j in dist_feature:
        cat_dict[count].append(j)
        cat_data.append((feature ==j).astype(int))
cat_data = np.array(cat_data).T
cat_data = np.array(cat_data, dtype='int')
print(cat_dict)

cat_data_test = []
non_cat_data_test = data_test[:,non_cat]
for count, i in enumerate(cat):
    feature = data_test[:, i]
    dist_feature = set(feature)
    for j in cat_dict[count]:
        cat_data_test.append((feature ==j).astype(int))
cat_data_test = np.array(cat_data_test).T
cat_data_test = np.array(cat_data_test, dtype='int')


# zip categorical and non-categorical data together
train_data = np.concatenate((cat_data, non_cat_data), axis=1)
train_label = data[:, -1].astype(int)
validation_data = train_data[:6000, :]
validation_label = train_label[:6000]
train_data = train_data[6000:,:]
train_label = train_label[6000:]
test_data = np.concatenate((cat_data_test,non_cat_data_test), axis=1)

# plot accuracy
accuracy = []
for i in range(40):
    tree = DecisionTree(i,105)
    tree.train(train_data,train_label)
    res = tree.predict(validation_data)
    score = 0
    for i in range(len(res)):
        if res[i] == validation_label[i]:
            score += 1
    score /= len(res)
    accuracy.append(score)

plt.plot(accuracy)
plt.xlabel('depth')
plt.ylabel('accuracy')
plt.title('Accuracy vs. Decision Tree Depth')
plt.savefig('p6.png')
plt.show()



# decision tree

tree = DecisionTree(10, train_data.shape[0])
tree.train(train_data, train_label)
res = tree.predict(validation_data[:1,:])


score = 0
for i in range(len(res)):
    if res[i] == validation_label[i]:
        score += 1
score /= len(res)
print(score)


# random forest

rf = RandomForest(10,10,train_data.shape[0],train_data.shape[1])
rf.train(train_data,train_label)
res = rf.predict(validation_data)
score = 0
for i in range(len(res)):
    if res[i] == validation_label[i]:
        score += 1
score /= len(res)
print(score)

# with open('titanic_prediction.csv', 'wt') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerow(['Id', 'Category'])
#     for i, cat in enumerate(res):
#         writer.writerow([str(i + 1), str(cat)])

