from decisionTree import DecisionTree
import numpy as np


class RandomForest:
    def __init__(self, num_trees, max_depth, num_sample, num_feature):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.num_sample = num_sample
        self.num_feature = num_feature
        self.trees = []

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        for i in range(self.num_trees):
            sample_index = np.random.choice(self.data.shape[0], self.num_sample, replace=True)
            train_data = self.data[sample_index, :]
            train_labels = self.labels[sample_index]
            tree = DecisionTree(self.max_depth,self.num_feature)
            tree.train(train_data, train_labels)
            self.trees.append(tree)


    def predict(self, data):
        labels = []
        for tree in self.trees:
            labels.append(tree.predict(data))
        res = np.mean(np.array(labels),axis=0)
        for i in range(len(res)):
            if res[i] >= 0.5:
                res[i] = 1
            else:
                res[i] = 0
        return res

