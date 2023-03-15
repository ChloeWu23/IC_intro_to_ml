##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import pandas as pd
from helper import *
from DecisionTreeWithPruning import DecisionTreeWithPruning

def train_and_predict(x_train, y_train, x_test, x_val, y_val):
    """ Interface to train and test the new/improved decision tree.

    This function is an interface for training and testing the new/improved
    decision tree classifier.

    x_train and y_train should be used to train your classifier, while
    x_test should be used to test your classifier.
    x_val and y_val may optionally be used as the validation dataset.
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K)
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K)
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K)
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )

    Returns:
    numpy.ndarray: A numpy array of shape (M, ) containing the predicted class label for each instance in x_test
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################

    # TODO: Train new classifier
    # set initial range of hyperparameters
    max_depth = list(range(8, 16))
    min_sample_split = list(range(5, 20, 2))
    min_sample_leaf = list(range(2, 12, 2))

    # Perform grid search and store the accuracy and classifier for each K 
    gridsearch_accuracies = []
    for depth in max_depth:
        for min_split in min_sample_split:
            for min_leaf in min_sample_leaf:
                tree = DecisionTreeWithPruning(depth, min_split, min_leaf)
                rootNode = tree.fit(x_train, y_train)
                y_predict = tree.predict(x_val)
                acc = accuracy(y_val, y_predict)
                gridsearch_accuracies.append([acc, depth, min_split, min_leaf, tree])

    da = pd.DataFrame(gridsearch_accuracies,
                      columns=["accuracy", "max_depth", "min_sample_split", "min_sample_leaf", "decision_tree"])

    y_pred_by_mode = np.zeros(len(x_test), dtype=y_val.dtype)
    Y_predicts = []

    # select trees with top 10% performance against validation set in terms of accuracy
    for tree in da.sort_values(by='accuracy', ascending=False).iloc[:int(len(gridsearch_accuracies)*0.1),-1]:
        Y_predicts.append(tree.predict(x_test))                                    
    
    # for each test sample, loop through the prediction from different trees to find the 'mode' class label
    for sample_i in range(len(y_pred_by_mode)):
        d_freq = dict.fromkeys(['A', 'C', 'E', 'G', 'O', 'Q'], 0)
        for tree_j in range(len(Y_predicts)):
            d_freq[Y_predicts[tree_j][sample_i]] += 1
        y_pred_by_mode[sample_i] = max(d_freq, key = d_freq.get)                         
                                             
    # remember to change this if you rename the variable
    return y_pred_by_mode
