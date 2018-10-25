import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.Classifier.RandomForest_Classifier import *

'''
Prerequisite: Database should be built before running demo.
The training data cannot have multiple labels.
Please speak loudly in a quiet place when you build the database.
'''

foldername = '../DataSet/DataList_all.txt'

'''
The performance of randomforest is worse than svm
When we choose to use zcr, the validating result is around 45%
and if we switch it to energy, the validating result is around 40%
'''

Classifier = RandomForest_Classifier(foldername)
Data, Label = Classifier.read_data(foldername, 'e', 25)
Classifier.train(Data, Label)
