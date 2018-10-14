import numpy as np


class FeatureExtractors:

    @staticmethod
    def enhance_frame(wav_data, frame_size, overlap, coefficient=0.97, windowing_method=None):
        length = wav_data.shape[0]
        step = frame_size - overlap
        #frame_num = np.cast(np.ceil(length / step), np.int)
        frame_num = np.ceil(length / step).astype(np.int)
        frame_data = np.zeros((frame_size, frame_num))

        for i in range(frame_num):
            """
            Reinforce the high-frequency parts of the voice here.
            System equation: y[t] = x[t] - coe * x[t - 1]
            """
            single_frame = wav_data[np.arange(i * step, min(i * step + frame_size, length))]
            single_frame = np.append(single_frame[0], single_frame[:-1] - coefficient * single_frame[1:])  # 预加重
            frame_data[:len(single_frame), i] = single_frame
            frame_data[:, i] = FeatureExtractors.windowing(windowing_method, frame_size, frame_data[:, i])

        return frame_data

    @staticmethod
    def zero_crossing_rate(frames):
        frame_num = frames.shape[1]
        frame_size = frames.shape[0]
        zcr = np.zeros((frame_num, 1))
        for i in range(frame_num):
            frame = frames[:, i]
            temp = frame[:-1] * frame[1:]
            temp = np.sign(temp)
            zcr[i] = np.sum(temp < 0)

        return zcr

    @staticmethod
    def energy(frames):
        frame_num = frames.shape[1]
        e = np.zeros((frame_num, 1))
        for i in range(frame_num):
            frame = frames[:, i]
            e[i] = np.sum(frame * frame)

        return e

    @staticmethod
    def windowing(method, size, frame):
        method_dict = {
            'Hamming': np.hamming,
            'Rectangular': np.ones,
        }
        if method not in method_dict.keys():
            return frame
        windowing_func = method_dict[method]
        return windowing_func(size) * frame
