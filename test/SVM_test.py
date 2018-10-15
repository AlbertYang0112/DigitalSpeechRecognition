import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.SVM_Classifier import *

foldername = '../DataSet/DataList.txt' 

Classifier = SVM_Classifier(foldername)
Data, Label = Classifier.read_data(foldername, 'zcr',25)
print(np.shape(Data))
print(Label)
Classifier.train(Data, Label)
print(Classifier.apply(Data))
print(Label)