import numpy as np
import wave


class FeatureExtractors:

    @staticmethod
    def enhance_frame(wav_data, frame_size, overlap, coefficient=0.97):
        length = len(wav_data)
        step = frame_size - overlap
        frame_num = np.ceil(length / step)
        frame_data = np.zeros((frame_size, frame_num))

        window = np.hamming(frame_size)

        for i in range(frame_num):
            """
            Reinforce the high-frequency parts of the voice here.
            System equation: y[t] = x[t] - coe * x[t - 1]
            """
            single_frame = wav_data[np.arange(i * step, min(i * step + frame_size, length))]
            single_frame = np.append(single_frame[0], single_frame[:-1] - coefficient * single_frame[1:])  # 预加重
            frame_data[:len(single_frame), i] = single_frame
            frame_data[:, i] = window * frame_data[:, i]  # 加窗

        return frame_data
