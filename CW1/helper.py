# helper functions for data examination
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy import stats

'''
Helper functions for data processing
'''
def read_data(path_to_file):
    """
    Helper function to read data from a given file

    Args:
    path (str): path to the data file

    Returns:
    1. a NumPy array of shape (N, K) representing N training instances of K attributes;
    2. a NumPy array of shape (N,) containing the class label for each N instance. The class label
    should be a string representing the character, e.g. "A", "E".

    """
    data_x = []
    data_y = []
    with open(path_to_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() != '':
                temp = line.strip().split(",")
                data_x.append(list(map(int, temp[:-1])))
                data_y.append(temp[-1])

    data = np.array(data_x)
    classlabel = np.array(data_y)

    return data, classlabel


def label_analyse(label, filename, subplot):
    """
    Helper function to count appearances per class label and plot it in a line chart

    Args:
    label (np.array) a NumPy array of shape (N,) containing the class label for each N instance
    filename (str) a string containing the filename, to be used as the legend in the plot
    subplot plt.subplot subplot objects to hold the plotted graph

    Returns:
    dictionary containing the class label, number of data with this label and its percentage in the total sample
    """
    data = {"letter": [], "count": [], "proportion": []}

    for letter in np.unique(label):
        count = np.sum(label == letter)
        pct = count / label.shape[0]

        data["letter"].append(letter)
        data["count"].append(count)
        data["proportion"].append(pct)
        print(f"{letter} appeared {count} times, {round(pct,2)} of the total set")

    subplot.plot(data["letter"], data["proportion"], label=filename)
    subplot.legend()

    return data, subplot


"""
Helper functions for model evaluation
"""
def confusion_matrix(labels, pred_labels, class_labels=None):
     """ Compute the confusion matrix.

     Args:
         labels (np.ndarray): the correct ground truth/gold standard labels
         pred_labels (np.ndarray): the predicted labels
         class_labels (np.ndarray): a list of unique class labels.
                                Defaults to the union of labels and pred_labels.

     Returns:
         np.array : shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions
     """

     # if no class_labels are given, we obtain the set of unique class labels from
     # the union of the ground truth annotation and the prediction
     if not class_labels:
         class_labels = np.unique(np.concatenate((labels, pred_labels)))

     confusion = np.zeros(
         (len(class_labels), len(class_labels)), dtype=int)

     # for each correct class (row),
     # compute how many instances are predicted for each class (columns)
     for (i, label) in enumerate(class_labels):
         # get predictions where the ground truth is the current class label
         indices = (labels == label)
         real = labels[indices]
         predictions = pred_labels[indices]

         # quick way to get the counts per label
         (unique_labels, counts) = np.unique(
             predictions, return_counts=True)

         # convert the counts to a dictionary
         frequency_dict = dict(zip(unique_labels, counts))

         # fill up the confusion matrix for the current row
         for (j, class_label) in enumerate(class_labels):
             confusion[i, j] = frequency_dict.get(class_label, 0)

     return confusion

def accuracy(labels, pred_labels):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        labels (np.ndarray): the correct ground truth/gold standard labels
        pred_labels (np.ndarray): the predicted labels

    Returns:
        float : the accuracy
    """

    assert len(labels) == len(pred_labels)

    try:
        return round(np.sum(labels == pred_labels) / len(labels), 5)
    except ZeroDivisionError:
        return 0.

def precision(labels, pred_labels):
    """ Compute the precision score per class given the ground truth and predictions

    Also return the macro-averaged precision across classes.

    Args:
        labels (np.ndarray): the correct ground truth/gold standard labels
        pred_labels (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the 
              precision for class c
            - macro-precision is macro-averaged precision (a float) 
    """

    confusion = confusion_matrix(labels, pred_labels)
    p = np.zeros((len(confusion),))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])

    macro_p = 0.
    if len(p) > 0:
        macro_p = np.mean(p)

    return (p, macro_p)

def recall(labels, pred_labels):
    """ Compute the recall score per class given the ground truth and predictions

    Also return the macro-averaged recall across classes.

    Args:
        labels (np.ndarray): the correct ground truth/gold standard labels
        pred_labels (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the
                recall for class c
            - macro-recall is macro-averaged recall (a float)
    """

    confusion = confusion_matrix(labels, pred_labels)
    r = np.zeros((len(confusion),))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])

    # Compute the macro-averaged recall
    macro_r = 0.
    if len(r) > 0:
        macro_r = np.mean(r)

    return (r, macro_r)

def f1_score(labels, pred_labels):
    """ Compute the F1-score per class given the ground truth and predictions

    Also return the macro-averaged F1-score across classes.

    Args:
        labels (np.ndarray): the correct ground truth/gold standard labels
        pred_labels (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float)
    """

    (precisions, macro_p) = precision(labels, pred_labels)
    (recalls, macro_r) = recall(labels, pred_labels)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions),))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    # Compute the macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)

    return (f, macro_f)

"""
Helper functions for cross validation
"""
def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a
            numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple)
            with two elements: a numpy array containing the train indices, and another
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])

        folds.append([train_indices, test_indices])

    return folds


def train_val_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple)
            with three elements:
            - a numpy array containing the train indices
            - a numpy array containing the val indices
            - a numpy array containing the test indices
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test, and k+1 as validation (or 0 if k is the final split)
        test_indices = split_indices[k]
        val_indices = split_indices[(k + 1) % n_folds]

        # concatenate remaining splits for training
        train_indices = np.zeros((0,), dtype=int)
        for i in range(n_folds):
            # concatenate to training set if not validation or test
            if i not in [k, (k + 1) % n_folds]:
                train_indices = np.hstack([train_indices, split_indices[i]])

        folds.append([train_indices, val_indices, test_indices])

    return folds

"""
Class for decision tree nodes
"""

class Node:
    """
    Class for decision tree nodes
    Attributes:
        leftNode: reference to the left of child's Node
        rightNode: reference to right of child's Node
        feature: integer from 0 to 15, indicate which feature number it is.
        splitPoint: current Node's split value of this feature.
        X: training sets of features, N*K 2D arralabels
        labels: training sets of labels , N*1 arralabels
        isLeaf(bool): whether this node is leaf or not
        classLabel(char): when it is a leaf, return the label of the class
    """

    def __init__(self, feature=-1, X=None, labels=None, splitPoint=-1):
        """
        Initialize object in class Node
        """
        self.leftNode = None
        self.rightNode = None
        self.feature = feature
        self.X = X
        self.labels = labels
        self.isLeaf = False
        self.splitPoint = splitPoint
        self.classLabel = None

    def addChild(self, childNode):
        """
        Function to add Child Nodes
        @param childNode: an object of Node which to be added
        """
        if (self.leftNode == None):
            self.leftNode = childNode
            return
        if (self.rightNode == None):
            self.rightNode = childNode
            return
        print("LeftNode and RightNode are Full, Can not add new Node")



