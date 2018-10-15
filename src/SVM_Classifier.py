from sklearn import svm
import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.PreProcessing import *
from src.VoiceDataSetBuilder import *
from src.FileLoader import *
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

class SVM_Classifier:
    '''
    This is the SVM_Classifier of different voice information.
    We try to utilized multiple methods to actualize the classification task
    and this is one of them.
    '''
    def __init__(self, DataListName):
        self.DataListName = '../DataSet/DataList.txt'
        self.clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')

    def read_data(self, DataListName, FeatureName,shape):
        '''
        Load original data and the feature that you choose to use is needed
        You can choose different output shape.
        :DataListName: The log.txt path
        :FeatureName: The name of the feature that you choose to use
        :shape: The output shape that you prefer
        '''
        Feature = FeatureName.capitalize()
        Data = np.zeros((1,shape))
        Label = []
        processer = PreProcessing(512, 128)
        wav_list, frame_list, energy_list, zcr_list, endpoint_list, Label = processer.process(DataListName)
        if Feature[0] == 'E':
            for i in range(len(zcr_list)):
                temp = processer.effective_feature(energy_list[i], endpoint_list[i])
                temp = processer.reshape(temp, shape)
                Data=np.concatenate((Data,temp),axis = 0)
            Data = Data[1:]
            return Data, Label
        elif Feature[0] == 'Z':
            for i in range(len(zcr_list)):
                temp = processer.effective_feature(zcr_list[i], endpoint_list[i])
                temp = processer.reshape(temp, shape)
                Data=np.concatenate((Data,temp),axis = 0)
            Data = Data[1:]
            return Data, Label
        else:
            print("please choose correct feature, and we will return ZCR by default")
            for i in range(len(zcr_list)):
                temp = processer.effective_feature(zcr_list[i], endpoint_list[i])
                temp = processer.reshape(temp, shape)
                Data=np.concatenate((Data,temp),axis = 0)
            Data = Data[1:]
            return Data, Label
        
    def train(self, Data, Label):
        '''
        Train SVM model
        Feature data and labels are needed
        '''
        clf = self.clf
        x_train = Data
        y_train = Label
        '''
        The database is not big enough to be splited.
        When database is big enougth you can choose to split original database
        and set validation data.
        '''
        #x_train, x_test, y_train, y_test = train_test_split(Data, Label, random_state=1, train_size=0.8)
        clf.fit(x_train, y_train)  # svm classification
        print("training result")
        print(clf.score(x_train, y_train))  # svm score
        y_hat = clf.predict(x_train)
        print("validating result")
        #print(clf.score(x_test, y_test))
        #y_hat = clf.predict(x_test)
        joblib.dump(clf, "train_model.m")
    
    def apply(self, Data):
        '''
        Apply a model to predict
        '''
        clf = self.clf
        clf = joblib.load("train_model.m")
        return clf.predict(Data)

    def show_accuracy(self, y_pre, y_true, Signal):
        print(Signal)
    