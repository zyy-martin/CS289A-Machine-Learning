import numpy as np
import csv
import pandas as pd
import collections
from sklearn.feature_extraction import DictVectorizer
from decisionTree import DecisionTree
from randomForest import RandomForest

# move the first column to the last

# new_data = []
# with open('hw5_titanic_dist/titanic_training.csv') as f:
#     reader = csv.reader(f, delimiter=',')
#     for line in reader:
#         new_data.append(line)
# new_data = np.array(new_data)
# new_data = np.concatenate((new_data[:,1:], new_data[:,0].reshape((new_data[:,0].shape[0],1))), axis=1)
#
# with open('hw5_titanic_dist/titanic_training_new.csv','wt') as f:
#     writer = csv.writer(f, delimiter=',')
#     for line in new_data:
#         writer.writerow(line)


data_test = []
data = []
dict = {}

# load training data

with open('hw5_titanic_dist/titanic_training_new.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        if i == 0:
            for j in range(len(row) - 1):
                dict[j] = row[j]
        else:
            data.append(row)
# fill in missing data
data = np.array(data)
most_common_dict = {}
for i in range(data.shape[1] - 1):
    most_common_dict[i] = collections.Counter(data[:, i]).most_common(1)[0][0]
sums = 0
count = 0
for i in range(data.shape[0]):
    if data[i, 2] == '':
        continue
    else:
        sums += float(data[i, 2])
        count += 1
most_common_dict[2] = sums / count
print(most_common_dict)

for i in range(data.shape[0]):
    for j in range(data.shape[1] - 1):
        if data[i, j] == '':
            data[i, j] = most_common_dict[j]

# load testing data

with open('hw5_titanic_dist/titanic_testing_data.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        if i == 0:
            continue
        else:
            data_test.append(row)
data_test = np.array(data_test)
print(most_common_dict)

# fill in missing data

for i in range(data_test.shape[0]):
    for j in range(data_test.shape[1]):
        if data_test[i, j] == '':
            data_test[i, j] = most_common_dict[j]

# preprocess data

# print(dict)
non_cat = [2, 3, 4, 6]
cat = [0, 1, 8]

cat_dict = []
cat_data = []
non_cat_data = data[:, non_cat]
non_cat_data = np.array(non_cat_data, dtype='float')

for count, i in enumerate(cat):
    cat_dict.append([])
    feature = data[:, i]
    dist_feature = set(feature)
    for j in dist_feature:
        cat_dict[count].append(j)
        cat_data.append((feature == j).astype(int))
cat_data = np.array(cat_data).T
cat_data = np.array(cat_data, dtype='float')
print(cat_dict)

non_cat_data_test = data_test[:, non_cat]
non_cat_data_test = np.array(non_cat_data_test, dtype='float')
cat_data_test = []

for count, i in enumerate(cat):
    feature = data_test[:, i]
    dist_feature = set(feature)
    for j in cat_dict[count]:
        cat_data_test.append((feature == j).astype(int))
cat_data_test = np.array(cat_data_test).T
cat_data_test = np.array(cat_data_test, dtype='float')

# zip categorical and non-categorical data together

train_data = np.concatenate((cat_data, non_cat_data), axis=1)
train_label = data[:, -1].astype(int)
validation_data = train_data[:200, :]
validation_label = train_label[:200]
train_data = train_data[:, :]
train_label = train_label[:]
test_data = np.concatenate((cat_data_test, non_cat_data_test), axis=1)


# decision tree
tree = DecisionTree(5, train_data.shape[0])
tree.train(train_data, train_label)
res = tree.predict(validation_data)
score = 0
for i in range(len(res)):
    if res[i] == validation_label[i]:
        score += 1
score /= len(res)
print(score)


# random forest

forest = RandomForest(100,5,train_data.shape[0],6)
forest.train(train_data, train_label)
res = forest.predict(validation_data)

score = 0
for i in range(len(res)):
    if res[i] == validation_label[i]:
        score += 1
score /= len(res)
print(score)


# write to csv
# with open('titanic_prediction.csv', 'wt') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerow(['Id', 'Category'])
#     for i, cat in enumerate(res):
#         writer.writerow([str(i + 1), str(cat)])
