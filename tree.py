# to run the code input file in line 29 
# put X = all x values as numpy ndarray
# put Y = labels as (0,1,2,3,4)  NOT ONE HOT
# In line 40 Give name to features

from sklearn import tree
from sklearn.tree import _tree
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def getrules(tree, features):
    tree_ = tree.tree_
    feature = [features[i] if i != _tree.TREE_UNDEFINED else -2 for i in tree_.feature]
    print ("Rule Tree")

    def recurse(node, depth):
        indent = "\t" * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            print(indent," if ",feature[node]," <= ",tree_.threshold[node],":")
            recurse(tree_.children_left[node], depth + 1)
            print(indent,"else:")
            recurse(tree_.children_right[node], depth + 1)
        else:
            print(indent,"return ",np.nonzero(tree_.value[node][0])[0][0])

    recurse(0, 1)

df = pd.read_csv("./abc.data", sep=',',header=None)
X = np.array(df[[0,1,2]])
Y = np.array(df[3])

Xtr, Xts, Ytr, Yts = train_test_split(X, Y,test_size=100)


clf = tree.DecisionTreeClassifier(criterion = "entropy",max_depth=4)
clf = clf.fit(Xtr, Ytr)
Ypd = clf.predict(Xts)  
print(clf.score(Xts,Yts))
getrules(clf,["x1","x2","x3"])
print(confusion_matrix(Yts, Ypd))  
print(classification_report(Yts, Ypd))
print(clf.predict_proba(Xts))