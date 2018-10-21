import numpy as np
import keras
from keras.layers import *
from sklearn.ensemble import RandomForestClassifier
import sys
import os
from keras.utils import np_utils
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.PreProcessing import *
from src.VoiceDataSetBuilder import *
from src.FileLoader import *
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.models import Model

class CNN_Classifier:
    
    def __init__(self,DataListName):
        self.DataListName = DataListName
    
    def read_data(self, DataListName,FeatureName,shape):
        '''
        Load original data and the feature that you choose to use is needed
        You can choose different output shape.
        :DataListName: The log.txt path
        :FeatureName: The name of the feature that you choose to use
        :shape: The output shape that you prefer
        '''
        Feature = FeatureName.capitalize()
        Data = np.zeros((1,shape))
        zcrdata = np.zeros((1,shape))
        energydata = np.zeros((1,shape))
        Label = []
        processer = PreProcessing(512, 128)
        wav_list, frame_list, energy_list, zcr_list, endpoint_list, Label = processer.process(DataListName)
        if Feature[0] == 'E':
            for i in range(len(energy_list)):
                temp = processer.effective_feature(energy_list[i], endpoint_list[i])
                temp = processer.reshape(temp, shape)
                if len(temp) == 0:
                    Label = Label[0:i-1]+Label[i:]
                    continue
                Data=np.concatenate((Data,temp),axis = 0)
            Data = Data[1:]
            return Data, Label
        elif Feature[0] == 'Z':
            for i in range(len(zcr_list)):
                temp = processer.effective_feature(zcr_list[i], endpoint_list[i])
                temp = processer.reshape(temp, shape)
                if len(temp) == 0:
                    Label = Label[0:i-1]+Label[i:]
                    continue
                Data=np.concatenate((Data,temp),axis = 0)
            Data = Data[1:]
            print(np.shape(Data))
            print(np.shape(Label))
            return Data, Label
        elif Feature[0] == 'W':
            for i in range(len(zcr_list)):
                temp = processer.effective_feature(zcr_list[i], endpoint_list[i])
                temp = processer.reshape(temp, shape)
                if len(temp) == 0:
                    Label = Label[0:i-1]+Label[i:]
                    continue
                zcrdata = np.concatenate((zcrdata,temp),axis = 0)
            zcrdata = zcrdata[1:]
            print(np.shape(zcrdata))
            for i in range(len(zcr_list)):
                temp = processer.effective_feature(energy_list[i], endpoint_list[i])
                temp = processer.reshape(temp, shape)
                if len(temp) == 0:
                    continue
                energydata =np.concatenate((energydata,temp),axis = 0)
            energydata = energydata[1:]
            data = energydata * zcrdata
            return data, Label
        else:
            print("please choose correct feature, and we will return ZCR by default")
            for i in range(len(zcr_list)):
                temp = processer.effective_feature(zcr_list[i], endpoint_list[i])
                temp = processer.reshape(temp, shape)
                if len(temp) == 0:
                    Label = Label[0:i-1]+Label[i:]
                    continue
                Data=np.concatenate((Data,temp),axis = 0)
            Data = Data[1:]
            return Data, Label
    
    def data_processer(self, Data, Label, shape, nb_classes):
        X_train, X_test, Y_train, Y_test = train_test_split(Data, Label, random_state=0, train_size=0.75)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= np.max(np.abs(X_train))
        X_test /= np.max(np.abs(X_test))
        X_train = np.expand_dims(X_train, 0)
        X_train = X_train.reshape(X_train.shape[1], shape,1)
        X_test = X_test.reshape(X_test.shape[0], shape,1)
        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(Y_test, nb_classes)
        input_shape = np.shape(X_train)
        print("INPUT SHAPE = ", input_shape)
        return X_train, X_test, Y_train, Y_test, input_shape
    
    def cnn_model(self,input_shape, pool_size, nb_filters, kernel_size, nb_classes):
        model = Sequential()
        model.add(Convolution1D(nb_filters, kernel_size,
                                padding='same',input_shape = input_shape[1:])) # 卷积层1
        model.add(Activation('relu')) #激活层
        model.add(MaxPooling1D(pool_size=pool_size)) #池化层
        model.add(Convolution1D(nb_filters, 10)) #卷积层2
        model.add(Activation('relu')) #激活
        model.add(Convolution1D(nb_filters, 10)) #卷积层2
        model.add(Activation('relu')) #激活
        model.add(Convolution1D(nb_filters, 10)) #卷积层2
        model.add(Activation('relu')) #激活
        model.add(Convolution1D(nb_filters, 10)) #卷积层2
        model.add(Activation('relu')) #激活
        model.add(Convolution1D(nb_filters, 10)) #卷积层2
        model.add(Activation('relu')) #激活
        model.add(Convolution1D(nb_filters, 10,
                                padding='same')) # 卷积层1
        model.add(Activation('relu')) #激活层
        model.add(MaxPooling1D(pool_size=5)) #池化层

        #model.add(ConvLSTM2D(128,kernel_size = kernel_size, dropout=0.5, recurrent_dropout=0.5))
        #model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dropout(0.5)) #神经元随机失活
        model.add(Flatten()) #拉成一维数据
        model.add(Dense(256)) #全连接层1
        model.add(Activation('relu')) #激活层
        model.add(Dropout(0.75)) #随机失活
        model.add(Dense(128)) #全连接层1
        model.add(Activation('relu')) #激活层
        model.add(Dropout(0.25)) #随机失活
        model.add(Dense(nb_classes)) #全连接层
        model.add(Activation('softmax')) #激活层
        model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])
        return model
    
    def train(self, model, X_train, X_test, Y_train, Y_test, nb_classes, epochs, batch_size, log_dir):
        model.fit(X_train, Y_train, batch_size = batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test), callbacks=[TensorBoard(log_dir= log_dir, histogram_freq=1,write_graph=True)])
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
