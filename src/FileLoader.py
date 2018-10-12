import wave
import numpy as np


class FileLoader:
    """
    Class Fileloader
    Feed it with a log file containing path to the data and relating label;
    Run read_next() to iterate through the data set.
    Log file format:
        PathToData Label
    """

    def __init__(self, data_list):
        self.MUTE_LIMIT = 0.05  # Any signal below this threshold will be ignored

        self.__dataset = []
        with open(data_list, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                filePath, label = line.split(' ')
                self.__dataset.append((filePath, label))
        print("Dataset Size:", len(self.__dataset))
        self.__data_pointer = 0

    def read_next(self):
        file_path = self.__dataset[self.__data_pointer][0]
        label = self.__dataset[self.__data_pointer][1]
        with wave.open(file_path, 'rb') as wavFile:
            params = wavFile.getparams()
            channels, sample_width, frame_rate, frame_num = params[:4]
            str_data = wavFile.readframes(frame_num)
            wav_data = np.fromstring(str_data, dtype=np.int16)
            wav_min, wav_max = np.min(wav_data), np.max(wav_data)
            wav_data = (wav_data - wav_min) / (wav_max - wav_min)   # Normalize the wave
            wav_data = np.where(np.abs(wav_data) < self.MUTE_LIMIT,
                                0, wav_data)

        if self.__data_pointer >= len(self.__dataset):
            self.__data_pointer = 0
        else:
            self.__data_pointer += 1

            return wav_data, label
