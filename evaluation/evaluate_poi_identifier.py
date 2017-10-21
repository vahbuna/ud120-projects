#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import tree
from sklearn.model_selection import train_test_split

clf = tree.DecisionTreeClassifier()

features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

#28. Number of POIs in Test Set
print len([i for i in labels_test if i==1.0])

#29. Number of People in Test Set
print len(labels_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score
predictions =  [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
#34. True positives
print tp

#35. True Negatives
print tn

#36. False Positives
print fp

#37. False Negatives
print fn

#38. Precision
precision = tp*1.0/(tp + fp)
print precision, precision == precision_score(true_labels, predictions)

#39. Recall
recall = tp*1.0/(tp+fn)
print recall, recall == recall_score(true_labels, predictions)
