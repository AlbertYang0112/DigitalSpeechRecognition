import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.Classifier.DecisionTree_Classifier import *

'''
Prerequisite: Database should be built before running demo.
The training data cannot have multiple labels.
Please speak loudly in a quiet place when you build the database.
'''

foldername = '../DataSet/DataList.txt' 

'''
This is the demo of decision tree classifier for comparative trial
and the best performance is around 54%
'''

Classifier = DecisionTree_Classifier(foldername)
Data, Label = Classifier.read_data(foldername, 'w',25)
Classifier.train(Data, Label)
