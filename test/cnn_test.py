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

def read_data(DataListName, FeatureName, shape):
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
        return Data, Label
    elif Feature[0] == 'W':
        for i in range(len(zcr_list)):
            temp = processer.effective_feature(wav_list[i], endpoint_list[i])
            temp = processer.reshape(temp, shape)
            if len(temp) == 0:
                Label = Label[0:i-1]+Label[i:]
                continue
            Data=np.concatenate((Data,temp),axis = 0)
        Data = Data[1:]
        return Data, Label
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

foldername = '../DataSet/DataList.txt'
shape = 290
Data, Label = read_data(foldername, 'zcr', shape)

X_train, X_test, Y_train, Y_test = train_test_split(Data, Label, random_state=0, train_size=0.7)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(np.abs(X_train))
X_test /= np.max(np.abs(X_test))

batch_size = 128
nb_classes = 10
epochs = 700
pool_size = 10
nb_filters = 50
kernel_size = 10
# 固定随机数种子以复现结果
seed=13
np.random.seed(seed)

# 创建 1 维向量，并扩展维度适应 Keras 对输入的要求， data_1d 的大小为 (1, 25, 1)
X_train = np.expand_dims(X_train, 0)
print(np.shape(X_train))
X_train = X_train.reshape(X_train.shape[1], shape,1)
X_test = X_test.reshape(X_test.shape[0], shape,1)
print(np.shape(X_train))
print(np.shape(X_test))
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
input_shape = np.shape(X_train)
# 编译模型
model = Sequential()
"""
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=input_shape))
"""
model.add(Convolution1D(nb_filters, kernel_size,
                        padding='same',input_shape = input_shape[1:])) # 卷积层1
model.add(Activation('relu')) #激活层
model.add(MaxPooling1D(pool_size=pool_size)) #池化层
model.add(Convolution1D(nb_filters, 10)) #卷积层2
model.add(Activation('relu')) #激活
model.add(Convolution1D(nb_filters, 10)) #卷积层2
model.add(Activation('relu')) #激活
model.add(Convolution1D(nb_filters, 20,
                        padding='same')) # 卷积层1
model.add(Activation('relu')) #激活层
model.add(MaxPooling1D(pool_size=5)) #池化层
#model.add(LSTM(128, dropout=0.25, recurrent_dropout=0.25))
#model.add(LSTM(256, dropout=0.25, recurrent_dropout=0.25))
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
#训练模型
model.fit(X_train, Y_train, batch_size = batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))
#评估模型
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])