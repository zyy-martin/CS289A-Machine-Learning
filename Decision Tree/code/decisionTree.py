import numpy as np
import random
import collections


class Node:
    def __init__(self, split_rule, left, right, label, is_leaf):
        self.split_rule = split_rule
        self.left = left
        self.right = right
        self.label = label
        self.is_leaf = is_leaf

    def get_split_rule(self):
        return self.split_rule

    def get_left_child(self):
        return self.left

    def get_right_child(self):
        return self.right

    def is_leaf(self):
        return self.is_leaf

    def get_label(self):
        return self.label


class DecisionTree:
    def __init__(self, max_depth, num_features, measure='entropy'):
        self.max_depth = max_depth
        self.num_features = num_features
        self.measure = measure
        self.root = Node(None, None, None, None, False)

    def cost(self, prob):
        if self.measure == 'entropy':
            if float(prob) == 0.0:
                return 0
            elif float(prob) == 1.0:
                return 0
            else:
                return - prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)
        elif self.measure == 'polynomial':
            if prob == 0.0:
                return 0
            elif prob == 1.0:
                return 0
            else:
                return (1 - prob) * prob
        else:
            print('Error. No method found')
            return None

    def impurity(self, left_label_hist, right_label_hist):
        left_size = sum(left_label_hist)
        right_size = sum(right_label_hist)
        if left_size == 0:
            right_prob = right_label_hist[0] / right_size
            right_cost = self.cost(right_prob)
            return right_cost
        elif right_size == 0:
            left_prob = left_label_hist[0] / left_size
            left_cost = self.cost(left_prob)
            return left_cost
        else:
            left_prob = left_label_hist[0] / left_size
            left_cost = self.cost(left_prob)
            right_prob = right_label_hist[0] / right_size
            right_cost = self.cost(right_prob)
            return (left_size * left_cost + right_size * right_cost) / (left_size + right_size)



    def segmenter(self, data, labels):
        threshold_list = (np.mean(data[labels == 1], axis=0) + np.mean(data[labels == 0], axis=0)) / 2
        feature_collection = random.sample(range(len(data[0])), self.num_features)
        impurity_list = []
        for i in feature_collection:
            threshold = threshold_list[i]
            group1_labels = labels[data[:, i] >= threshold]
            group2_labels = labels[data[:, i] < threshold]
            left_labels_hist = [len(group1_labels) - sum(group1_labels), sum(group1_labels)]
            right_labels_hist = [len(group2_labels) - sum(group2_labels), sum(group2_labels)]
            impurity_list.append(self.impurity(left_labels_hist, right_labels_hist))
        best_feature_index = feature_collection[impurity_list.index(min(impurity_list))]
        best_feature_threshold = threshold_list[best_feature_index]
        # print('best feature is: ',best_feature_index, ', best threshold is: ',best_feature_threshold)
        return best_feature_index, best_feature_threshold





    def grow_tree(self, data, labels, height):
        # print('depth of this node: ',height)
        if height <= self.max_depth:
            if len(np.unique(labels)) == 1:
                # print('pure leaf: ',np.unique(labels)[0])
                return Node(None, None, None, np.unique(labels)[0], True)
            # elif len(labels) < 10:
            #     common_label = collections.Counter(labels).most_common(1)[0][0]
            #     return Node(None, None, None, common_label, True)
            else:
                [index, threshold] = self.segmenter(data, labels)
                left = data[:, index] < threshold
                right = data[:, index] >= threshold
                left_data = data[left]
                right_data = data[right]
                left_label = labels[left]
                right_label = labels[right]
                if len(left_label) == 0 or len(right_label) == 0:
                    common_label = collections.Counter(labels).most_common(1)[0][0]
                    # print('none leaf: ', common_label)
                    return Node(None, None, None, common_label, True)
                else:
                    left_node = self.grow_tree(left_data, left_label, height + 1)
                    right_node = self.grow_tree(right_data, right_label, height + 1)
                    return Node((index, threshold), left_node, right_node, None, False)

        else:
            common_label = collections.Counter(labels).most_common(1)[0][0]
            # print('reach max: ',common_label)
            return Node(None, None, None, common_label, True)


    def traverse(self, root, x):
        if root.is_leaf == True:
            print('reach leaf', root.label)
            return root.label
        else:
            index = root.split_rule[0]
            threshold = root.split_rule[1]

            if x[index] <= threshold:
                print('feature: ',index, ' <= ',threshold )
                return self.traverse(root.left, x)
            else:
                print('feature: ', index, ' >', threshold)
                return self.traverse(root.right, x)



    def train(self, data, labels):
        self.root = self.grow_tree(data, labels, 1)
        print(self.root.split_rule)

    def predict(self, data):
        predict_res = []
        for x in data:
            predict_res.append(self.traverse(self.root, x))
        return predict_res








