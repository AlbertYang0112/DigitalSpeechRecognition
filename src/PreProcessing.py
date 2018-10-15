# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from src.FileLoader import FileLoader
from src.FeatureExtractors import FeatureExtractors
from scipy import interpolate


class PreProcessing:
    def __init__(self, frame_size, overlap):
        self.frame_size = frame_size
        self.overlap = overlap
        self.loader = FileLoader()
        plt.ion()
        plt.figure(2)

    def process(self, file_name):
        file_name = file_name.strip()
        if file_name[-3:] == 'wav':
            wav_data = self.loader.read_one(file_name)
            frames = FeatureExtractors.enhance_frame(
                wav_data=wav_data,
                frame_size=self.frame_size,
                overlap=self.overlap,
                windowing_method='Hamming'
            )
            energy = FeatureExtractors.energy(frames)
            zcr = FeatureExtractors.zero_crossing_rate(frames)
            endpoint = self.VAD_advance(energy)
            return wav_data, frames, energy, zcr, endpoint
        elif file_name[-3:] == 'txt':
            self.loader.set_data_list(file_name)
            wav_list = []
            frame_list = []
            energy_list = []
            zcr_list = []
            endpoint_list = []
            label_list = []
            for i in range(self.loader.data_set_size()):
                wav_data, label = self.loader.read_next()
                frames = FeatureExtractors.enhance_frame(
                    wav_data=wav_data,
                    frame_size=self.frame_size,
                    overlap=self.overlap,
                    windowing_method='Hamming'
                )
                energy = FeatureExtractors.energy(frames)
                zcr = FeatureExtractors.zero_crossing_rate(frames)
                endpoint = self.VAD_advance(energy)
                zcr = np.reshape(zcr, [len(zcr)])
                endpoint = np.reshape(endpoint, [len(endpoint)])
                wav_list.append(wav_data)
                frame_list.append(frames)
                energy_list.append(energy)
                label_list.append(label)
                zcr_list.append(zcr)
                endpoint_list.append(endpoint)
            return wav_list, frame_list, energy_list, zcr_list, endpoint_list, label_list

    # 新增的利用双门限法的语音端点检测
    # 增强了识别能力，可以用于多数字的语音信息的断点识别
    def VAD_advance(self, energy):
        MEAN = np.sum(energy) / len(energy)
        High = 0.8 * MEAN  # 语音能量上限
        Low = 0.02 * MEAN  # 能量下限
        Data1 = []  # 存放低位能量数据
        Data2 = []  # 存放高位能量数据
        Endpoint = []  # 存放两个节点
        Flag = 1  # 状态位
        Flag2 = 1  # 状态位
        i = 1
        if energy[i-1] > Low:
            Data1.append(i-1)
            Flag = 0
        while i < len(energy):
            if energy[i] > Low and Flag == 1: #当能量高于低阈值时
                if energy[i-1] < Low: #如果上一帧的能量低于低阈值
                    Data1.append(i-1) #将此节点记录下来
                    Flag = 0 #状态位置零
            if energy[i] > High and Flag2 == 1: #当能量高于高阈值时
                if energy[i-1] < High: #如果上一帧的能量低于高阈值
                    Data2.append(i-1) #将此节点记录下来
                    Flag2 = 0 #状态位置零
            if energy[i] < Low and Flag == 0: #当能量低于低阈值时
                if energy[i-1] > Low: #如果上一帧能量高于低阈值
                    Data1.append(i-1) #将此节点记录下来
                    Flag = 1 #状态位置一
            if energy[i] < High and Flag2 == 0: #当能量低于高阈值时
                if energy[i-1] > High: #如果上一帧能量高于高阈值
                    Data2.append(i-1) #将此节点记录下来
                    Flag2 = 1 #状态位置于一
            i += 1
        i = 0
        j = 0
        if len(Data2) == 0:
            print("No active voice detected")
            return []
        
        while j < int(len(Data2) / 2):  # 循环遍历数据点，筛选有效的节点
            i = 0
            while i < int(len(Data1) / 2):
                if Data1[2 * i] <= Data2[2 * j] and Data1[2 * i + 1] >= Data2[2 * j + 1]:
                    Endpoint.append(Data1[2 * i])
                    Endpoint.append(Data1[2 * i + 1])
                    break
                i += 1
            j += 1
        counter = 2
        temp = 0
        i = 2
        while i < len(Endpoint) - 2:  # 消除重复点
            if Endpoint[i] != Endpoint[i - 2]:
                Endpoint[counter] = Endpoint[i]
                counter += 1
            i += 1
        if len(Endpoint) == 0:
            print("No active voice detected")
            return []
        else:
            endpoint = []
            for i in range(counter):
                endpoint.append(Endpoint[i])
            return endpoint

    @staticmethod
    def effective_feature(features, endpoints):
        feature_list = []
        segment_num = int(len(endpoints) / 2)
        for i in range(segment_num):
            start, end = int(endpoints[2 * i]), int(endpoints[2 * i + 1])
            feature_list.append(features[start: end])
        return feature_list

    @staticmethod
    def reshape(Data, shape):
        data_set = []
        for i in range(len(Data)):
            index = len(Data[i])
            if index < 5:
                continue
            new_shape = np.linspace(0, index, shape)
            data = np.reshape(Data[i], index)
            x = np.linspace(0, index, index)
            f = interpolate.interp1d(x, data, kind='cubic')
            data_set.append(f(new_shape))
        return data_set

    def print_result(self, filename, frame, energy, zcr, endpoint):
        '''
        :param filename: file name of the signal
        :param frame: result of enframe
        :param energy: energy of signal after enframing
        :param zcr: zcr of signal for each frame
        :param endpoint: the endpoint of the signal judged by energy
        :return: none
        '''
        data = self.wavread(filename)
        plt.figure(1)
        # plot the voice data
        plt.subplot(221)
        plt.grid('on')
        plt.plot(data[0])
        plt.axvline(endpoint[0] * (512 - 128), color='r')
        plt.axvline(endpoint[1] * (512 - 128), color='r')
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.title("Signal")

        # plot the result of enframe
        plt.subplot(222)
        plt.grid('on')
        plt.plot(frame[0])
        plt.axvline(endpoint[0], color='r')
        plt.axvline(endpoint[1], color='r')
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.title("Enframe")

        # plot the energy of signal
        plt.subplot(223)
        plt.grid('on')
        plt.plot(energy)
        plt.axvline(endpoint[0], color='r')
        plt.axvline(endpoint[1], color='r')
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.title("Energy")

        # plt the information of zcr
        plt.subplot(224)
        plt.grid('on')
        plt.plot(zcr)
        plt.axvline(endpoint[0], color='r')
        plt.axvline(endpoint[1], color='r')
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.title("zcr")
        plt.grid('on')

        # show the plot result
        plt.show()
