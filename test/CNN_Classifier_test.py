import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.Classifier.CNN_Classifier import *
from keras.models import Model

'''
This is the demo program of cnn classifier.
The structure of this convolutional neural network is shown below
########################################
INPUT_SHAPE = (NUMBER_OF_DATA, SHAPE, 1)
Conv1D Activation = ReLU
Max_pooling
Conv1D Activation = ReLU
Conv1D Activation = ReLU
Conv1D Activation = ReLU
Max_pooling
Dropout = 0.5
Fully_Connected 256 Activation = ReLU
Fully_Connected 128 Activation = ReLU
Fully_Connected 10 Activation = SoftMax
Loss = catagroical_crossentropy
########################################
Prerequisite:
Keras 2.1
TensorFlow-GPU
TensorBoard
'''

shape = 290
batch_size = 200
nb_classes = 10
epochs = 1000
pool_size = 3
nb_filters = 50
kernel_size = 3

foldername = '../DataSet/DataList.txt' 
'''
The log file will be saved in the folder ./tmp/log.
Using TensorBoard, you can find the visual data in 
training process and the structure of this cnn.
'''
log_dir = './tmp/log'

Classifier = CNN_Classifier(foldername)
Data, Label = Classifier.read_data(foldername, 'w', shape)
X_train, X_test, Y_train, Y_test, input_shape = Classifier.data_processer(Data, Label, shape, nb_classes)
model = Classifier.cnn_model(input_shape, pool_size, nb_filters, kernel_size, nb_classes)
Classifier.train(model, X_train, X_test, Y_train, Y_test, nb_classes, epochs, batch_size, log_dir)
Classifier.get_mid_data(model)