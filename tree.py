from sklearn import tree
from sklearn.tree import _tree
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

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

df = pd.read_csv("./data_150593.csv")
X = np.array(df[["x1","x2","x3"]])
Y = np.array(df["class"])

Xtr, Xts, Ytr, Yts = train_test_split(X, Y,test_size=20)



clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtr, Ytr)
print(clf.score(Xts,Yts))
getrules(clf,["x1","x2","x3"])