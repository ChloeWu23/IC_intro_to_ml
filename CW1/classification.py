#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from helper import Node

class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    rootNode(node): the root node of the decision tree, default by None
    Methods:
    fit(x, y): Constructs a decision tree from data X and label Y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """
    
    def __init__(self, node = None):
        if node is not None:
            self.__is_trained = True
            self.__rootNode = node
            return
        self.__is_trained = False
        self.__rootNode = None
        
    
    def __entropy(self, labels):
        """
        compute entropy according to the labels
        labels: an array with elements in the range of 0-5
        return: entropy
        """
        #prob = np.zeros(len(labels),dtype=float)
        (label, count) = np.unique(labels, return_counts = True)
        prob = count / np.sum(count)
        return np.sum(-prob * (np.log2(prob)))
    

    def __IG(self, allLabel, leftLabel, rightLabel):
        """ 
        Function to calculate Information Gain
        allLabel: the labels of data in parent node , N*1 array
        leftLabel: subset of allLabel, labels of data in left node
        rightLabel: subset of allLabel, labels of data in right node
        return: Information Gain
        """
        if (allLabel.size != (leftLabel.size + rightLabel.size)):
            return None
        entropy1 = self.__entropy(allLabel)
        p = leftLabel.size / allLabel.size
        entropy2 = p * self.__entropy(leftLabel) + (1-p)*self.__entropy(rightLabel)
        return entropy1 - entropy2
        
    def __isLeaf(self, X, labels):
        ''' 
        Check the given node is Leaf or not,
         X: N * K np array
         labels: N*1 np array
         return boolean 
        '''
        return (np.unique(labels).size == 1 or X.shape[1] == 1 or np.allclose(X, X[0]))


    def __find_best_split_point(self, X, labels):
        """ Function to find the best Split feature Number with largest infromation Gain
            featureNum: how many features are in the training sets, 16
            X: N*K
            labels: N*1
            return: Node
        """
        result = {} #key is a pair of (i,splitPoint)
        featureNum = X.shape[1]
        
        for i in range(featureNum):
            #calculate information gain accoding to this feature's split
            for splitPoint in range(np.min(X[:,i]).astype(int), np.max(X[:,i]).astype(int)):
                leftLabel = labels[X[:,i] <= splitPoint]
                rightLabel = labels[X[:,i] > splitPoint]
                #calculate IG for each split
                result[(i, splitPoint)] = self.__IG(labels, leftLabel, rightLabel)
            
        #find max key in the dictionary:
        max_key = max(result, key = result.get) 
        #print(max_key)

        return Node(feature = max_key[0], splitPoint = max_key[1])

    def __split_node(self, X, labels, parentNode): 
        ''' Function to split data with parent node
            labels: N*1 array 	
            X: N*k array
            parentNode(Node)
            return : splited data according to the split point
        '''
        
        assert(X.shape[0] == len(labels)), "Wrong Input shape"
        
        feature = parentNode.feature
        #splitPoint need to change later
        splitPoint = parentNode.splitPoint
        X1 = X[X[:,feature] <= splitPoint]
        label1 = labels[X[:,feature] <= splitPoint]
        #print(sublabel1)
        X2 = X[X[:,feature] > splitPoint]
        label2 = labels[X[:,feature] > splitPoint]
        return [(X1, label1), (X2,label2)]
        
        
    
    def __construct_tree(self, X, labels, depth = 0):
        """  
        recursively construct the tree
        return a tree Structure node
        """
        if self.__isLeaf(X, labels) or depth > len(labels):
            uniq, counts = np.unique(labels, return_counts = True)
            node = Node()
            node.isLeaf = True
            node.X = X
            node.labels = labels
            node.classLabel = uniq[np.argmax(counts)]
            return node
        
        #if not the leaf, split the data into two parts
        parentNode = self.__find_best_split_point(X, labels)
        parentNode.X = X
        parentNode.labels = labels
        childDataSet = self.__split_node(X, labels, parentNode)
        #print(childDataSet[0][1])
        for childData in childDataSet:
            childNode = self.__construct_tree(childData[0], childData[1], depth+1)
            # childNode.X = childData[0]
            # childNode.labels = childData[1]
            parentNode.addChild(childNode)
            
        return parentNode


        
    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
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
        #self.__is_trained = True
        #max_deepth = len(x.shape[1]) 
        #find the best feature for root Node split
        self.__rootNode = self.__construct_tree(x, y)
        self.__is_trained = True
        return self.__rootNode

    def __predict_one_sample(self, node, sample):
        """ 
        sample: 1*K np array with all the features
        return its predicted label
        """
        
        if node.isLeaf:
            return node.classLabel

        if (sample[node.feature] <= node.splitPoint):
            return self.__predict_one_sample(node.leftNode, sample)
        else:
            return self.__predict_one_sample(node.rightNode, sample)
        


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
        if not self.__is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        # self.is_trained = False
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype= object)
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        
        for i in range(x.shape[0]):
            predictions[i] = self.__predict_one_sample(self.__rootNode, x[i, :])
           
        # remember to change this if you rename the variable
        return predictions
    
    def __print_node(self, node, depth=0):
        dash = "--"*depth
        print(f"{dash}{node.feature}_{node.splitPoint}_{len(node.X)}")
        if node.leftNode is not None:
            self.__print_node(node.leftNode, depth+1)
        if node.rightNode is not None:
            self.__print_node(node.rightNode, depth+1)
        
    def print_tree(self):
        self.__print_node(self.__rootNode, 0)
        
        
