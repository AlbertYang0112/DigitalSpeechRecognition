import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.Classifier.SVM_Classifier import *

'''
Prerequisite: Database should be built before running demo.
The training data cannot have multiple labels.
Please speak loudly in a quiet place when you build the database.
'''

foldername = '../DataSet/DataList.txt' 

'''
the performance in zcr is much better than energy
the validating result of zcr can reach 50% but result of energy can only reach 46% with
low training score around 75%.
'''

Classifier = SVM_Classifier(foldername)
Data, Label = Classifier.read_data(foldername, 'z',15)
Classifier.train(Data, Label)
