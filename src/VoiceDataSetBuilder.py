import os
from datetime import datetime
from src.Recorder import Recorder
import matplotlib.pyplot as plt
import pyaudio


class VoiceDataSetBuilder(Recorder):
    """
    Build the data set.
    """

    def __init__(self, dst_path, rec_length=2, log_file=None, display=True,
                 chunk=1024, audio_format=pyaudio.paInt16, channels=1, rate=8000):
        """
        Initialize the procedure.
        :param dst_path: the directory to contain the data set.
                        If the directory do not exist, then the
                        function will automatically create one
        :param rec_length: specify the length of wav file
        :param log_file: the full path to the log file of the data set.
                        Log format: path label
                        The path above is relative path
        :param display: display the wave just recorded if True
        :param chunk: The same as class Recorder
        :param audio_format: The same as class Recorder
        :param channels: The same as class Recorder
        :param rate: The same as class Recorder
        """
        super(VoiceDataSetBuilder, self).__init__(chunk, audio_format, channels, rate)
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
        self.PATH = dst_path
        self.LENGTH = rec_length
        self.DISPLAY = display
        self.LOG = log_file

    def generate_name(self, label):
        return label + '_' + str(self.get_datetime_without_space()) + '.wav'

    def get_datetime_without_space(self):
        return str(datetime.now()).replace(' ', '-')

    def append_log(self, full_filename, label):
        if self.LOG is None:
            return
        if os.path.exists(self.LOG):
            with open(self.LOG, 'a') as log_file:
                log_file.write(full_filename + ' ' + label + '\n')
        else:
            with open(self.LOG, 'w') as log_file:
                log_file.write(full_filename + ' ' + label + '\n')

    def build(self):
        """
        Execute the building procedure
        :return: a tuple. The first element is a list containing
                all the 1-D-numpy-array wave in the data set, and
                the second one is a list containing the corresponding label.
        """
        if self.DISPLAY:
            plt.ion()
            plt.figure(1)
        rec_finished = False
        rec_wav_list = []
        rec_label = []
        while not rec_finished:
            label = input('Input Label(Number):')
            file_name = self.generate_name(label)
            full_path = os.path.join(self.PATH, file_name)
            print("Recording", file_name)
            print("Recording Start: ")
            wav = self.rec_one_shot(sec=self.LENGTH, file_name=None)
            if self.DISPLAY:
                plt.clf()
                plt.title(file_name)
                plt.plot(wav)
                plt.draw()
                plt.pause(0.1)

            invalid_cmd = True
            while invalid_cmd:
                print("\nC->Continue  S->Save&Stop  R->Redo  E->Exit without save\nPlease input(Letter):")
                cmd = input().capitalize()
                if len(cmd) == 0:
                    continue

                if cmd[0] == 'C':
                    print("Continue to the next\n")
                    self.save_wav(wav, full_path)
                    self.append_log(full_path, label)
                    rec_wav_list.append(wav)
                    rec_label.append(label)
                    invalid_cmd = False
                elif cmd[0] == 'E':
                    print("Exit without save it\n")
                    rec_finished = True
                    invalid_cmd = False
                elif cmd[0] == 'S':
                    print("Save it and Exit\n")
                    rec_finished = True
                    invalid_cmd = False
                    self.save_wav(wav, full_path)
                    self.append_log(full_path, label)
                    rec_wav_list.append(wav)
                    rec_label.append(label)
                elif cmd[0] == 'R':
                    print("Redo this record\n")
                    invalid_cmd = False
        return rec_wav_list, rec_label

