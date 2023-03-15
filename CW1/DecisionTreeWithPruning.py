import numpy as np
from helper import Node

class DecisionTreeWithPruning(object):
    """ decision tree with pruning, considering max_depth, min_sample_split, min_sample_leaf
    max_depth: the maximum depth the tree can grow
    min_sample_split: the minimal sample size of a node that can split
    min_sample_leaf: the minimal sample size required to be a leaf node

    Attributes:
    """

    def __init__(self, maxDepth, minSplit, minLeaf, node = None):
        if node is not None:
            self.is_trained = True
            self.rootNode = node
        else:
            self.is_trained = False
            self.rootNode = None
            self.maxDepth = maxDepth
            self.minSplit = minSplit
            self.minLeaf = minLeaf

    def entropy(self, labels):
        """
        compute entropy according to the labels
        labels: an array with elements in the range of 0-5
        return: entropy
        """
        # prob = np.zeros(len(labels),dtype=float)
        (label, count) = np.unique(labels, return_counts = True)
        prob = count / np.sum(count)
        return np.sum(-prob * (np.log2(prob)))

    def IG(self, allLabel, leftLabel, rightLabel):
        """
        Function to calculate Information Gain
        allLabel: the labels of data in parent node , N*1 array
        leftLabel: subset of allLabel, labels of data in left node
        rightLabel: subset of allLabel, labels of data in right node
        return: Information Gain
        """
        if (allLabel.size != (leftLabel.size + rightLabel.size)):
            return None
        entropy1 = self.entropy(allLabel)
        p = leftLabel.size / allLabel.size
        entropy2 = p * self.entropy(leftLabel) + ( 1 -p ) *self.entropy(rightLabel)
        return entropy1 - entropy2

    def isOriginLeaf(self, node):
        '''
        Check the given node is Leaf or not,
         X: N * K np array
         labels: N*1 np array
         return boolean
        '''

        return (np.unique(node.labels).size == 1 or node.X.shape[0] <= self.minLeaf)

    def makeLeaf(self, node):

        node.leftNode = None
        node.rightNode = None
        node.isLeaf = True
        node.feature = -1
        node.splitPoint = -1
        uniq, counts = np.unique(node.labels, return_counts=True)
        #print(uniq, counts)
        node.classLabel = uniq[np.argmax(counts)]

        return node

    def find_best_split_point(self, node):
        """ Function to find the best Split feature Number with largest infromation Gain
            featureNum: how many features are in the training sets, 16
            X: N*K
            labels: N*1
            return: Node
        """
        X = node.X
        labels = node.labels
        result = {}  # key is a pair of (i,splitPoint)
        featureNum = X.shape[1]

        for i in range(featureNum):
            # calculate information gain accoding to this feature's split

            for splitPoint in range(np.min(X[: ,i]).astype(int), np.max(X[: ,i]).astype(int)):
                leftLabel = labels[X[: ,i] <= splitPoint]
                rightLabel = labels[X[: ,i] > splitPoint]
                # calculate IG for each split
                result[(i, splitPoint)] = self.IG(labels, leftLabel, rightLabel)

        while True:

            # find max key in the dictionary:
            max_key = max(result, key = result.get)

            node.feature = max_key[0]
            node.splitPoint = max_key[1]

            X1 = X[X[:, node.feature] <= node.splitPoint]
            X2 = X[X[:, node.feature] > node.splitPoint]

            # if one of the child node's sample size is lower than minLeaf, then this split is invalid
            # repeat until find the key such that the sizes of both child nodes are qualified

            if (X1.shape[0] >= self.minLeaf and X2.shape[0] >= self.minLeaf):
                return node

            # delete this split, find the next max_key
            del result[max_key]
            if len(result) == 0:
                break

        # if didn't find the split point, make it a leaf node
        self.makeLeaf(node)
        return node

    def split_node(self, parentNode):
        ''' Function to split data with parent node
            labels: N*1 array
            X: N*k array
            parentNode(Node)
            return : splited data according to the split point
        '''
        X = parentNode.X
        labels = parentNode.labels
        assert(X.shape[0] == len(labels)), "Wrong Input shape"

        feature = parentNode.feature
        splitPoint = parentNode.splitPoint
        X1 = X[X[: ,feature] <= splitPoint]
        label1 = labels[X[: ,feature] <= splitPoint]
        X2 = X[X[: ,feature] > splitPoint]
        label2 = labels[X[: ,feature] > splitPoint]

        node1 = Node(X = X1, labels = label1)
        node2 = Node(X = X2, labels = label2)
        return node1, node2

    def construct_tree(self, node, depth = 0):
        """
        recursively construct the tree
        return a tree Structure node
        """

        if node.isLeaf:
            return node

        # if this is leaf node || sample size is lower than minSplit || the depth has reached maxDepth
        if self.isOriginLeaf(node) or node.X.shape[0] < self.minSplit or depth >= self.maxDepth:
            self.makeLeaf(node)
            return node


        # if this node can be splitted, split the data into two parts

        node = self.find_best_split_point(node)
        if node.isLeaf is not True:
            childNode1, childNode2 = self.split_node(node)

            node.addChild(self.construct_tree(childNode1, depth + 1))
            node.addChild(self.construct_tree(childNode2, depth + 1))

        return node

    def fit(self, x, y):
        """ Constructs a decision tree with pruning from data
        Algorithm: Recursive for right and left leaf from parent
        Args:
        x (numpy.nparray): Instances, numpy array of shape (N, K)
                           N is the number of instances
                           K is the number of attributes
        y (numpy.nparray): Class labels, numpy array of shape (N, )
                           Each element in y is a str
        return: Root Node of the tree
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################
        # set a flag so that we know that the classifier has been trained
        # self.is_trained = True
        # max_deepth = len(x.shape[1])
        # find the best feature for root Node split
        root = Node(X = x, labels = y)
        self.rootNode = self.construct_tree(root)
        self.is_trained = True
        return self.rootNode

    def predict_one_sample(self, node, sample):
        """
        sample: 1*K np array with all the features
        return its predicted label
        """

        if node.isLeaf:
            return node.classLabel

        if (sample[node.feature] <= node.splitPoint):
            return self.predict_one_sample(node.leftNode, sample)
        else:
            return self.predict_one_sample(node.rightNode, sample)

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of test instances
                           K is the number of attributes

        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x

        Algorithm:
        First for each row in x, find its Node according to value of its X,
        the predicted value should be max count of label in that Node
        """
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        # self.is_trained = False
        # set up an empty (M, ) numpy array to store the predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype= object)
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        for i in range(x.shape[0]):
            predictions[i] = self.predict_one_sample(self.rootNode, x[i, :])

        # remember to change this if you rename the variable
        return predictions