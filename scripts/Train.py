import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.SVM_Classifier import *
from src.RandomForest_Classifier import *

'''
Prerequisite: Database should be built before running demo.
The training data cannot have multiple labels.
Please speak loudly in a quiet place when you build the database.
'''

foldername = '../DataSet/DataList_ymr.txt'

Classifier = SVM_Classifier(foldername)

Data, Label = Classifier.read_data(foldername, 'zcr', shape=25)
#print(np.shape(Data))
#print(Label)
print(np.shape(Data))
print(np.shape(Label))
Classifier.train(Data, Label)

#print(Classifier.apply(Data))
#print(Label)
