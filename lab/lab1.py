import os
import numpy as np
import subprocess
import requests
#import data first
#data is from downloaded website
'''
if not os.path.exists("iris.data"):
    args = ["wget", "-O", "iris.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"]
subprocess.Popen(args)
'''
#another way to do import data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
r = requests.get(url,allow_redirects= True)
#save the file
open('iris.data','wb').write(r.content)

#read the data 
for line in open("iris.data"):
    print(line.strip()) #note strip() method is used for removing leading and tailing characters


'''
write function to lable data, x is to store 4 attributes of the data; y is used to show category of the data which is 0,1,2
class is an array with 3 elements to show how many categoris
'''
def read_data(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array. 
               - x is a numpy array with shape (N, K), 
                   where N is the number of instances
                   K is the number of features/attributes
               - y is a numpy array with shape (N, ), and each element should be 
                   an integer from 0 to C-1 where C is the number of classes 
               - classes : a numpy array with shape (C, ), which contains the 
                   unique class labels corresponding to the integers in y
    """
    x = []
    y_labels = []
    for line in open(filepath):
        if line.strip() != "":
            row = line.strip().split(",")
            x.append(list(map(float,row[:-1])))
            y_labels.append(row[-1])
            #note here y_labels are name of the category not number
    [classes,y]= np.unique(y_labels,return_inverse=True)
    #print(y)

    #If True, also return the indices of the unique array (for the specified axis, if provided) that can be used to reconstruct ar.
    x= np.array(x)
    y= np.array(y)
    return(x,y,classes)


(x,y,classes)= read_data("iris.data")
print(x.shape)
print(y.shape)
print(classes)

#understand the feature
print(x.min(axis=0))
print(x.max(axis=0))
print(x.mean(axis=0))
print(np.median(x, axis=0)) # for some reason x does not have a median() method
print(x.std(axis=0))

for class_label in np.unique(y):
    print("\nClass", class_label)
    x_class = x[y == class_label]
    print(x_class.min(axis=0))
    print(x_class.max(axis=0))
    print(x_class.mean(axis=0))    
    print(np.median(x_class, axis=0)) 
    print(x_class.std(axis=0))   

#plot 
import matplotlib.pyplot as plt

feature_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
                        
plt.figure()
plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.Set1, edgecolor='b')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.show()
#test
