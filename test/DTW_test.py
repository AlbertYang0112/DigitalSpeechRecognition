import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.Classifier.DTW_Classifier import *

'''
Prerequisite: Database should be built before running demo.
The training data cannot have multiple labels.
Please speak loudly in a quiet place when you build the database.
'''

foldername = '../DataSet/DataList_all.txt' 


'''
This is the demo of decision DTW classifier
and the best performance is around 85%
'''

Classifier = DTW_Classifier(foldername)
Data, Label = Classifier.read_data(foldername)
Classifier.train(Data, Label)
result = Classifier.apply(Data)
Classifier.show_accuracy(result, Label)