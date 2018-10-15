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

    def __init__(self, data_list=None):
        self.MUTE_LIMIT = 0.05  # Any signal below this threshold will be ignored
        self.__dataset = None
        self.__size = 0
        self.set_data_list(data_list)
        self.__data_pointer = 0

    def data_set_size(self):
        return self.__size


    def set_data_list(self, data_list):
        if data_list is None:
            self.__dataset = None
            return
        self.__dataset = []
        self.__size = 0
        with open(data_list, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                file_path, label = line.split(' ')
                self.__dataset.append((file_path, label))
                self.__size += 1
        print("Dataset Size:", self.__size)
        self.__data_pointer = 0

    def read_next(self):
        """
        Return the next wave array.
        The function cycles the data set specified in construction function.
        :return: Return a tuple, consists of a 1-D numpy array containing wave data
                    and a scalar of ground truth.
        """
        if self.__dataset is None:
            raise IOError
        file_path = self.__dataset[self.__data_pointer][0]
        label = self.__dataset[self.__data_pointer][1]
        with wave.open(file_path, 'rb') as wavFile:
            params = wavFile.getparams()
            channels, sample_width, frame_rate, frame_num = params[:4]
            str_data = wavFile.readframes(frame_num)
            wav_data = np.fromstring(str_data, dtype=np.int16)
            wav_data = wav_data * 1.0 / np.max(np.abs(wav_data))
            wav_data = np.where(np.abs(wav_data) < self.MUTE_LIMIT,
                                0, wav_data)

        if self.__data_pointer > len(self.__dataset):
            self.__data_pointer = 0
        else:
            self.__data_pointer += 1

            return wav_data, label

    def read_one(self, file_name):
        with wave.open(file_name, 'rb') as wavFile:
            params = wavFile.getparams()
            channels, sample_width, frame_rate, frame_num = params[:4]
            str_data = wavFile.readframes(frame_num)
            wav_data = np.fromstring(str_data, dtype=np.int16)
            wav_data = wav_data * 1.0 / np.max(np.abs(wav_data))
            wav_data = np.where(np.abs(wav_data) < self.MUTE_LIMIT,
                                0, wav_data)

        return wav_data
