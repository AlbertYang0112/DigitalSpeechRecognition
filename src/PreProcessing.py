# -*- coding:utf-8 -*-
import wave
import matplotlib.pyplot as plt
import numpy as np
import math

'''
demo:
filename = 'database/0.wav'
test = PreProcessing(512, 128)
frame, energy, zcr, endpoint = test.process(filename)
test.print_result(filename, frame, energy, zcr, endpoint)
'''


class PreProcessing:
    def __init__(self, framesize, overlap):
        self.framesize = framesize
        self.overlap = overlap

    def process(self, filename):
        '''
        :param filename: the name of signal file
        :return: frame. energy, zcr, endpoint
        '''
        data = self.wavread(filename)
        frame = self.Enframe(data[0])
        energy = self.energy(frame)
        zcr = self.ZCR(frame)
        endpoint = self.VAD_advance(energy)
        return frame, energy, zcr, endpoint

    def wavread(self, filename):
        f = wave.open(filename, 'rb')
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        f.close()
        waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
        waveData = np.reshape(waveData, [nframes, nchannels]).T
        return waveData

    def Enframe(self, wavData):  # 分帧加窗函数
        frameSize, overlap = self.framesize, self.overlap
        coeff = 0.97  # 预加重系数
        wlen = len(wavData)
        step = frameSize - overlap
        frameNum = int(math.ceil(wlen / step))
        frameData = np.zeros((frameSize, frameNum))

        hamwin = np.hamming(frameSize)

        for i in range(frameNum):
            singleFrame = wavData[np.arange(i * step, min(i * step + frameSize, wlen))]
            singleFrame = np.append(singleFrame[0], singleFrame[:-1] - coeff * singleFrame[1:])  # 预加重
            frameData[:len(singleFrame), i] = singleFrame
            frameData[:, i] = hamwin * frameData[:, i]  # 加窗，汉明窗，可以改

        return frameData

    # 计算每一帧的过零率
    def ZCR(self, frameData):
        frameNum = frameData.shape[1]  # 获取分帧阵的形态
        frameSize = frameData.shape[0]
        zcr = np.zeros((frameNum, 1))  # 设置一个空的矩阵

        for i in range(frameNum):
            singleFrame = frameData[:, i]  # 分别对每一帧内的数据进行操作
            temp = singleFrame[:frameSize - 1] * singleFrame[1:frameSize]  # 对相邻的位进行相乘操作
            temp = np.sign(temp)  # 将结果转化为符号
            zcr[i] = np.sum(temp < 0)  # 将负数个数求总数
        return zcr

    # 计算每一帧能量
    def energy(self, frameData):
        frameNum = frameData.shape[1]
        ener = np.zeros((frameNum, 1))
        for i in range(frameNum):
            singleframe = frameData[:, i]
            ener[i] = sum(singleframe * singleframe)
        return ener

    # 新增的利用双门限法的语音端点检测
    # 增强了识别能力，可以用于多数字的语音信息的断点识别
    def VAD_advance(self, energy):
        MEAN = np.sum(energy) / len(energy)
        High = 0.8 * MEAN  # 语音能量上限
        Low = 0.015 * MEAN  # 能量下限
        Data1 = []  # 存放低位能量数据
        Data2 = []  # 存放高位能量数据
        Endpoint = []  # 存放两个节点
        Flag = 1  # 状态位
        Flag2 = 1  # 状态位
        for i in range(len(energy)):
            if energy[i] > Low and Flag == 1:  # 当能量高于低阈值时
                if energy[i - 1] < Low:  # 如果上一帧的能量低于低阈值
                    Data1.append(i - 1)  # 将此节点记录下来
                    Flag = 0  # 状态位置零
            if energy[i] > High and Flag2 == 1:  # 当能量高于高阈值时
                if energy[i - 1] < High:  # 如果上一帧的能量低于高阈值
                    Data2.append(i - 1)  # 将此节点记录下来
                    Flag2 = 0  # 状态位置零
            if energy[i] < Low and Flag == 0:  # 当能量低于低阈值时
                if energy[i - 1] > Low:  # 如果上一帧能量高于低阈值
                    Data1.append(i)  # 将此节点记录下来
                    Flag = 1  # 状态位置一
            if energy[i] < High and Flag2 == 0:  # 当能量低于高阈值时
                if energy[i - 1] > High:  # 如果上一帧能量高于高阈值
                    Data2.append(i)  # 将此节点记录下来
                    Flag2 = 1  # 状态位置于一
        i = 0
        j = 0
        while j < len(Data2) / 2:  # 循环遍历数据点，筛选有效的节点
            i = 0
            while i < len(Data1) / 2:
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
            return 0
        else:
            endpoint = []
            for i in range(counter):
                endpoint.append(Endpoint[i])
            return endpoint

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


if __name__ == '__main__':
    filename = '../database/0.wav'
    test = PreProcessing(512, 128)
    frame, energy, zcr, endpoint = test.process(filename)
    test.print_result(filename, frame, energy, zcr, endpoint)
