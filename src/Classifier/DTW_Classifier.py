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

class DTW_Classifier:
    '''
    This is the DTW_Classifier of different voice information.
    We try to utilized multiple methods to actualize the classification task
    and this is one of them.
    '''

    def __init__(self, DataListName):
        '''
        The type of classifier should be choosed.
        '''
        self.DataListName = '../DataSet/DataList.txt'


    def read_data(self, DataListName):
        '''
        Load original data and the feature that you choose to use is needed
        You can choose different output shape.
        :DataListName: The log.txt path
        :FeatureName: The name of the feature that you choose to use
        :shape: The output shape that you prefer
        '''
        eff_label_list = []
        eff_mfcc = []
        processer = PreProcessing(512, 128)
        wav_list, frame_list, mfcc_list, energy_list, zcr_list, endpoint_list, label_list = processer.process(DataListName)
        for i in range(len(mfcc_list)):
            temp = processer.effective_feature(mfcc_list[i], endpoint_list[i])
            if endpoint_list[i][1]-endpoint_list[i][0] != 0:
                eff_label_list.append(label_list[i])
                eff_mfcc.append(mfcc_list[i])
            else:
                continue
        return eff_mfcc, eff_label_list
    
    def load_target(self, DataListName):
        eff_label_list = []
        eff_mfcc = []
        processer = PreProcessing(512, 128)
        wav_list, frame_list, mfcc_list, energy_list, zcr_list, endpoint_list, label_list = processer.process(DataListName)
        for i in range(len(mfcc_list)):
            temp = processer.effective_feature(mfcc_list[i], endpoint_list[i])
            if endpoint_list[i][1]-endpoint_list[i][0] != 0:
                eff_label_list.append(label_list[i])
                eff_mfcc.append(mfcc_list[i])
            else:
                continue
        return eff_mfcc, eff_label_list
        
    def classify(self, Data, Label, target_list, target_label_list):
        distance_list = []
        for i in range(len(Data)):
            for j in range(len(target_list)):
                distance_list.append(dtw(Data[i],target_list[j]))
            

    def dtw(self, x, y):
        """
        Computes Dynamic Time Warping (DTW) of two sequences.
        :param array x: N1*M array
        :param array y: N2*M array
        :param func dist: distance used as cost measure
        Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
        """
        assert len(x)
        assert len(y)
        r, c = len(x), len(y)
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
        D1 = D0[1:, 1:] # view
        for i in range(r):
            for j in range(c):
                D1[i, j] = linalg.norm(x[i] - y[j])
        C = D1.copy()
        for i in range(r):
            for j in range(c):
                D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
        if len(x)==1:
            path = zeros(len(y)), range(len(y))
        elif len(y) == 1:
            path = range(len(x)), zeros(len(x))
        else:
            path = _traceback(D0)
        return D1[-1, -1] / sum(D1.shape)
        #return D1[-1, -1] / sum(D1.shape), C, D1, path

    def _traceback(self, D):
        i, j = array(D.shape) - 2
        p, q = [i], [j]
        while ((i > 0) or (j > 0)):
            tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
            if (tb == 0):
                i -= 1
                j -= 1
            elif (tb == 1):
                i -= 1
            else: # (tb == 2):
                j -= 1
            p.insert(0, i)
            q.insert(0, j)
        return array(p), array(q)


    def show_accuracy(self, y_pre, y_true, Signal):
        print(Signal)
