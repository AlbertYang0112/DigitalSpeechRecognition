# -*- coding:utf-8 -*-

import wave
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as signal
# filepath = "" #添加路径
filename = 'database/0.wav'
# filename= os.listdir(filepath)  #得到文件夹下的所有文件名称
f = wave.open(filename, 'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)  # 读取音频，字符串格式
waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
waveData = waveData*1.0/(max(abs(waveData)))  # wave幅值归一化
Data = []
j = 0
for i in range(len(waveData)):
    if abs(waveData[i]) > 0.05:
        Data.append(waveData[i])
        j += 1
    
# plot the wave
time = np.arange(0,j)*(1.0 / framerate)
plt.plot(time,Data)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Single channel wavedata")
plt.grid('on')    # 标尺，on：有，off:无。


def Enframe(wavData, frameSize, overlap):  # 分帧加窗函数
    coeff = 0.97 # 预加重系数
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


def wavread(filename):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    f.close()
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    return waveData


# 计算每一帧的过零率
def ZCR(frameData):
    frameNum = frameData.shape[1]  # 获取分帧阵的形态
    frameSize = frameData.shape[0]
    zcr = np.zeros((frameNum, 1))  # 设置一个空的矩阵

    for i in range(frameNum):
        singleFrame = frameData[:, i]  # 分别对每一帧内的数据进行操作
        temp = singleFrame[:frameSize-1] * singleFrame[1:frameSize]  # 对相邻的位进行相乘操作
        temp = np.sign(temp)        # 将结果转化为符号
        zcr[i] = np.sum(temp < 0)   # 将负数个数求总数
    return zcr


# 计算每一帧能量
def energy(frameData):
    frameNum = frameData.shape[1]
    ener = np.zeros((frameNum, 1))
    for i in range(frameNum):
        singleframe = frameData[:, i]
        ener[i] = sum(singleframe * singleframe)
    return ener


# 新增的利用双门限法的语音端点检测
# 增强了识别能力，可以用于多数字的语音信息的断点识别
def VAD_advance(energy):
    MEAN = np.sum(energy)/len(energy)
    High = 0.8*MEAN  # 语音能量上限
    Low = 0.015*MEAN  # 能量下限
    Data1 = []  # 存放低位能量数据
    Data2 = []  # 存放高位能量数据
    Endpoint = []   # 存放两个节点
    Flag = 1  # 状态位
    Flag2 = 1  # 状态位
    for i in range(len(energy)):
        if energy[i] > Low and Flag == 1:  # 当能量高于低阈值时
            if energy[i-1] < Low:  # 如果上一帧的能量低于低阈值
                Data1.append(i-1)  # 将此节点记录下来
                Flag = 0  # 状态位置零
        if energy[i] > High and Flag2 == 1:  # 当能量高于高阈值时
            if energy[i-1] < High:  # 如果上一帧的能量低于高阈值
                Data2.append(i-1)  # 将此节点记录下来
                Flag2 = 0  # 状态位置零
        if energy[i] < Low and Flag == 0:  # 当能量低于低阈值时
            if energy[i-1] > Low:  # 如果上一帧能量高于低阈值
                Data1.append(i)  # 将此节点记录下来
                Flag = 1  # 状态位置一
        if energy[i] < High and Flag2 == 0:  # 当能量低于高阈值时
            if energy[i-1] > High:  # 如果上一帧能量高于高阈值
                Data2.append(i)  # 将此节点记录下来
                Flag2 = 1  # 状态位置于一
    i = 0
    j = 0
    while j < len(Data2)/2:  # 循环遍历数据点，筛选有效的节点
        i = 0
        while i < len(Data1)/2:
            if Data1[2*i] <= Data2[2*j] and Data1[2*i + 1] >= Data2[2*j + 1]:
                Endpoint.append(Data1[2*i])
                Endpoint.append(Data1[2*i+1])
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
    endpoint = []
    for i in range(counter):
        endpoint.append(Endpoint[i])
    return endpoint


if __name__ == '__main__':
    filename7 = "database/7.wav"
    filename0 = "database/0.wav"
    data0 = wavread(filename0)
    data7 = wavread(filename7)
    nw = 512
    inc = 128
    Frame0 = Enframe(data0[0], nw, inc)
    Frame7 = Enframe(data7[0], nw, inc)
    Energy0 = energy(Frame0)
    zcr0 = ZCR(Frame0)
    Energy7 = energy(Frame7)
    zcr7 = ZCR(Frame7)
    endpoint = VAD_advance(Energy0)
    endpoint2 = VAD_advance(Energy7)

    plt.figure(1)
    # plot the voice data
    plt.subplot(211)
    plt.grid('on')
    plt.plot(data0[0])
    plt.axvline(endpoint[0]*(512-128), color='r')
    plt.axvline(endpoint[1]*(512-128), color='r')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("data0")
    plt.subplot(212)
    plt.plot(data7[0])
    for i in range(len(endpoint2)):
        plt.axvline(endpoint2[i]*(512-128), color='r')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("data7")
    plt.grid('on')

    # plot the result of enframe
    plt.figure(2)
    plt.subplot(211)
    plt.grid('on')
    plt.plot(Frame0[0])
    plt.axvline(endpoint[0], color='r')
    plt.axvline(endpoint[1], color='r')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Data0")
    plt.subplot(212)
    plt.plot(Frame7[0])
    for i in range(len(endpoint2)):
        plt.axvline(endpoint2[i], color = 'r')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Data7")
    plt.grid('on')

    # plot the energy of signal
    plt.figure(3)
    plt.subplot(211)
    plt.grid('on')
    plt.plot(Energy0)
    plt.axvline(endpoint[0], color='r')
    plt.axvline(endpoint[1], color='r')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("energy0")
    plt.subplot(212)
    plt.grid('on')
    plt.plot(Energy7)
    for i in range(len(endpoint2)):
        plt.axvline(endpoint2[i], color='r')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("energy7")

    # plt the information of zcr
    plt.figure(4)
    plt.subplot(211)
    plt.grid('on')
    plt.plot(zcr0)
    plt.axvline(endpoint[0], color='r')
    plt.axvline(endpoint[1], color='r')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("zcr0")
    plt.grid('on')
    plt.subplot(212)
    plt.grid('on')
    plt.plot(zcr7)
    for i in range(len(endpoint2)):
        plt.axvline(endpoint2[i], color='r')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("zcr7")
    plt.grid('on')
    plt.show()
