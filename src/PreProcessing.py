# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from src.FileLoader import FileLoader
from src.FeatureExtractors import FeatureExtractors
from src.Recorder import Recorder
from scipy import interpolate
from multiprocessing import Queue, Process, set_start_method

TUNE_CHANGE_UPPER_BOUND = 1.5
TUNE_CHANGE_LOWER_BOUND = 0.75
TUNE_CHANGE_PROBABILITY = 0.5

VOLUME_CHANGE_UPPER_BOUND = 1.1
VOLUME_CHANGE_LOWER_BOUND = 0.9
VOLUME_CHANGE_PROBABILITY = 0.5

class PreProcessing:
    # Todo: Add the data argumentation methods.
    def __init__(self, frame_size, overlap):
        self.frame_size = frame_size
        self.overlap = overlap
        self.loader = FileLoader()
        self.recorder = Recorder()
        #plt.ion()
        #plt.figure(2)

    def process(self, file_name, argumentation=False):
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
            mfcc_feat = FeatureExtractors.mfcc_extractor(wav_data)
            return wav_data, frames, mfcc_feat, energy, zcr, endpoint
        elif file_name[-3:] == 'txt':
            label_count = [0 for i in range(10)]
            self.loader.set_data_list(file_name)
            wav_list = []
            frame_list = []
            mfcc_list = []
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
                if len(endpoint) == 0:
                    continue
                mfcc_feat = FeatureExtractors.mfcc_extractor(wav_data[endpoint[0] * (512 - 128) : endpoint[1] * (512 - 128)])
                zcr = np.reshape(zcr, [len(zcr)])
                endpoint = np.reshape(endpoint, [len(endpoint)])
                if label == '3' and np.random.random() < 0.1:
                    label_count[3] += 1
                    wav_list.append(wav_data)
                    frame_list.append(frames)
                    mfcc_list.append(mfcc_feat)
                    energy_list.append(energy)
                    label_list.append(label)
                    zcr_list.append(zcr)
                    endpoint_list.append(endpoint)

                label_count[int(label)] += 1

                wav_list.append(wav_data)
                frame_list.append(frames)
                mfcc_list.append(mfcc_feat)
                energy_list.append(energy)
                label_list.append(label)
                zcr_list.append(zcr)
                endpoint_list.append(endpoint)

                if argumentation:
                    argumentation_happened = False
                    if np.random.random() < TUNE_CHANGE_PROBABILITY:
                        tune_change_rate = np.random.random() * \
                                           (TUNE_CHANGE_UPPER_BOUND - TUNE_CHANGE_LOWER_BOUND) + TUNE_CHANGE_LOWER_BOUND
                        #print("Tune change:", wav_data.shape, tune_change_rate)
                        wav_data = self.reshape_1D(wav_data, tune_change_rate * wav_data.shape[0])
                        frames = FeatureExtractors.enhance_frame(
                            wav_data=wav_data,
                            frame_size=self.frame_size,
                            overlap=self.overlap,
                            windowing_method='Hamming'
                        )
                        #print("Tune Change!")
                        argumentation_happened = True

                    if np.random.random() < VOLUME_CHANGE_PROBABILITY:
                        frames = self.volume_argumentation(frames)
                        argumentation_happened = True
                        #print("Volume Change!")

                    if argumentation_happened:
                        #print("Write into!")
                        energy = FeatureExtractors.energy(frames)
                        zcr = FeatureExtractors.zero_crossing_rate(frames)
                        endpoint = self.VAD_advance(energy)
                        if len(endpoint) == 0:
                            continue
                        try:
                            mfcc_feat = FeatureExtractors.mfcc_extractor(wav_data[endpoint[0] * (512 - 128) : endpoint[1] * (512 - 128)])
                        except Exception as e:
                            print("Fucking Exception:", e)
                            print(endpoint)
                        zcr = np.reshape(zcr, [len(zcr)])
                        endpoint = np.reshape(endpoint, [len(endpoint)])
                        wav_list.append(wav_data)
                        frame_list.append(frames)
                        mfcc_list.append(mfcc_feat)
                        energy_list.append(energy)
                        label_list.append(label)
                        zcr_list.append(zcr)
                        endpoint_list.append(endpoint)

            print("Label Count:", label_count)
            return wav_list, frame_list, mfcc_list, energy_list, zcr_list, endpoint_list, label_list

    def process_stream(self):
        queue = self.recorder.stream_queue
        wav_queue = Queue()
        frame_queue = Queue()
        energy_queue = Queue()
        zcr_queue = Queue()
        endpoint_queue = Queue()
        mfcc_queue = Queue()
        queue_dict = {
            'wave': wav_queue,
            'frame': frame_queue,
            'energy': energy_queue,
            'zcr': zcr_queue,
            'endpoint': endpoint_queue,
            'mfcc': mfcc_queue
        }
        def conv_proc(wav_input, output_dict):
            PRE_FRAME_NUM = 10
            noise_energy = []
            for i in range(10):
                noise = queue.get(True).astype(np.float32)
                noise_energy.append(np.sum(noise * noise))
                print(noise)
            avg = np.average(noise_energy)
            variance = np.var(noise_energy)
            print('Energy:', noise_energy)
            print('Avg:', avg)
            print('Var:', variance)
            threshold = avg + 7 * np.sqrt(variance)
            print("THRESHOLD =", threshold)
            leading_frame = Queue(PRE_FRAME_NUM)
            recording = False
            state = 0
            tailing_cnt = 0
            rec = []
            cnt = 0
            while True:
                wav = wav_input.get(True).astype(np.float32)
                energy = np.sum(np.square(wav))
                cnt += 1

                if energy > threshold:
                    if state < 4:
                        state += 1
                    else:
                        recording = True
                        state = 20
                else:
                    if state > 0:
                        state -= 1
                    else:
                        if recording:
                            #plt.clf()
                            wav_full = np.concatenate(rec)
                            leading_frame_full = []
                            while not leading_frame.empty():
                                #print(leading_frame.qsize())
                                leading_frame_temp = leading_frame.get(True).astype(np.float32)
                                leading_frame_full.append(leading_frame_temp)
                            if len(leading_frame_full) != 0:
                                leading_frame_full = np.concatenate(leading_frame_full)
                                wav_full = np.concatenate((leading_frame_full, wav_full))
                                wav_full = wav_full * 1.0 / np.max(np.abs(wav_full))
                                wav_full = np.where(np.abs(wav_full) < 0.01,
                                                    0, wav_full)
                                #plt.subplot(121)
                                #plt.plot(wav_full)
                                #plt.subplot(122)
                                #plt.plot(leading_frame_full)
                                #plt.draw()
                                #plt.pause(0.001)
                                frames = FeatureExtractors.enhance_frame(
                                    wav_data=wav_full,
                                    frame_size=self.frame_size,
                                    overlap=self.overlap,
                                    windowing_method='Hamming'
                                )
                                energy = FeatureExtractors.energy(frames)
                                endpoint = self.VAD_advance(energy)
                                if len(endpoint) != 0:
                                    zcr = FeatureExtractors.zero_crossing_rate(frames)
                                    #mfcc = FeatureExtractors.mfcc_extractor(wav_full)
                                    mfcc = FeatureExtractors.mfcc_extractor(wav_full[endpoint[0] * (512 - 128) : endpoint[1] * (512 - 128)])
                                    zcr = np.reshape(zcr, [len(zcr)])
                                    endpoint = np.reshape(endpoint, [len(endpoint)])
                                    output_dict['wave'].put(wav_full)
                                    output_dict['frame'].put(frames)
                                    output_dict['energy'].put(energy)
                                    output_dict['zcr'].put(zcr)
                                    output_dict['endpoint'].put(endpoint)
                                    output_dict['mfcc'].put(mfcc)
                                    print("Fire!")
                                else:
                                    print("No EP")
                            else:
                                print("Empty??")
                        rec.clear()
                        recording = False
                if recording:
                    rec.append(wav)
                else:
                    if leading_frame.full():
                        leading_frame.get(True)
                    leading_frame.put(wav)

        proc = Process(target=conv_proc, args=(queue, queue_dict))
        return proc, queue_dict

    def process_continuous_stream(self):
        queue = self.recorder.stream_queue
        mfcc_queue = Queue()
        queue_dict = {
            'mfcc': mfcc_queue
        }
        def conv_proc(wav_input, output_dict):
            PRE_FRAME_NUM = 5
            noise_energy = []
            for i in range(10):
                noise = queue.get(True).astype(np.float32)
                noise_energy.append(np.sum(noise * noise))
                #print(noise)
            avg = np.average(noise_energy)
            variance = np.var(noise_energy)
            print('Energy:', noise_energy)
            print('Avg:', avg)
            print('Var:', variance)
            threshold = avg + 10 * np.sqrt(variance)
            print("THRESHOLD =", threshold)
            leading_frame = Queue(PRE_FRAME_NUM)
            recording = False
            state = 0
            tailing_cnt = 0
            rec = []
            cnt = 0
            while True:
                wav = wav_input.get(True).astype(np.float32)
                energy = np.sum(np.square(wav))
                cnt += 1
                #if cnt % 10 == 0:
                #    print("CNT:", cnt, energy, energy - threshold)

                if energy > threshold:
                    #print("Fire")
                    if state < 4:
                        state += 1
                        #print('Pre')
                    else:
                        recording = True
                        print('Voice Stream Start')
                        state = 8
                else:
                    if state > 0:
                        state -= 1
                        #print("Post", state)
                    else:
                        #print("Finish")
                        if recording:
                            print("Voice Stream Paused")
                        rec.clear()
                        recording = False
                if recording:
                    while not leading_frame.empty():
                        mfcc = FeatureExtractors.mfcc_extractor(leading_frame.get(True))
                        output_dict['mfcc'].put(mfcc)
                    mfcc = FeatureExtractors.mfcc_extractor(wav)
                    output_dict['mfcc'].put(mfcc)
                else:
                    if leading_frame.full():
                        leading_frame.get(True)
                    leading_frame.put(wav)

        proc = Process(target=conv_proc, args=(queue, queue_dict))
        return proc, queue_dict


    # Todo: Remove Chinese comments.
    # 新增的利用双门限法的语音端点检测
    # 增强了识别能力，可以用于多数字的语音信息的断点识别
    def VAD_advance(self, energy):
        energy = energy/np.max(energy)
        MEAN = 0.1
        High = MEAN  # 语音能量上限
        Low = 0.05*MEAN  # 能量下限
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
        elif Endpoint[1]-Endpoint[0]<3:
            print("Voice too short")
            return []
        else:
            endpoint = []   
            endpoint.append(Endpoint[0])
            endpoint.append(Endpoint[1])
            index = Endpoint[1] - Endpoint[0]
            for i in range(int(counter/2) -1):
                if Endpoint[2*i+1] - Endpoint[2*i] > index:
                    index = Endpoint[2*i+1] - Endpoint[2*i]
                    endpoint = []
                    endpoint.append(Endpoint[0])
                    endpoint.append(Endpoint[1])
            return endpoint

    @staticmethod
    def effective_feature(features, endpoints):
        feature_list = []
        segment_num = int(len(endpoints) / 2)
        for i in range(segment_num):
            start, end = int(endpoints[2 * i]), int(endpoints[2 * i + 1])
            feature_list.append(features[start: end])
        return np.array(feature_list)

    @staticmethod
    def reshape(Data, shape):
        data_set = []
        for i in range(len(Data)):
            index = len(Data[i])
            if index < 4:
                continue
            new_shape = np.linspace(0, index, shape)
            data = np.reshape(Data[i], index)
            x = np.linspace(0, index, index)
            f = interpolate.interp1d(x, data, kind='cubic')
            data_set.append(f(new_shape))
        return np.array(data_set)
    
    @staticmethod
    def reshape_1D(Data, shape):
        index = len(Data)
        new_shape = np.linspace(0, index, shape)
        data = np.reshape(Data, index)
        x = np.linspace(0, index, index)
        f = interpolate.interp1d(x, data, kind='cubic')
        return f(new_shape)

    @staticmethod
    def volume_argumentation(Data):
        '''
        This is data argumentation method for voice data which will randomly change the 
        volume of voice data by frame.
        :param Data: origin voice data
        :param rate: the index that determines the maximum change
        :return: changed voice data
        '''
        for i in range(len(Data)):
            Data[i] = Data[i] * \
                      (np.random.random() *
                       (VOLUME_CHANGE_UPPER_BOUND - VOLUME_CHANGE_LOWER_BOUND) + VOLUME_CHANGE_LOWER_BOUND)
        return Data

    
    def print_result(self, filename, frame, energy, zcr, endpoint):
        # Todo: Format the method.
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

    def __del__(self, exc_type, exc_val, exc_tb):
        plt.close()
