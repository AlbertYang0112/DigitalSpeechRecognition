import unittest
import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.PreProcessing import *
from src.VoiceDataSetBuilder import *
import matplotlib.pyplot as plt


class PreProcessingTests(unittest.TestCase):

    def setUp(self):
        self.DATA_DIR = './TEST_DIR'
        self.LOG = './TEST_DIR/data_log.txt'
        builder = VoiceDataSetBuilder(
            dst_path=self.DATA_DIR,
            log_file=self.LOG,
            rate=44100)
        builder.build()
        self.pre_process = PreProcessing(frame_size=512, overlap=128)
        plt.ion()
        plt.figure(1)

    def test_all(self):
        wav_list, frame_list, energy_list, zcr_list, endpoint_list = \
            self.pre_process.process(self.LOG)
        for i in range(len(frame_list)):
            wav_data = wav_list[i]
            print(np.max(np.abs(wav_data)))
            frames = frame_list[i]
            energys = energy_list[i]
            zcrs = zcr_list[i]
            endpoints = endpoint_list[i]
            effective_es = PreProcessing.effective_feature(zcrs, endpoints)
            plt.figure(i + 1)
            plt.subplot(221)
            plt.plot(wav_data)
            print(endpoints)
            for ep in endpoints:
                plt.axvline(ep * (self.pre_process.frame_size - self.pre_process.overlap), color='r')
            plt.subplot(222)
            plt.plot(energys)
            for ep in endpoints:
                plt.axvline(ep, color='r')
            plt.subplot(223)
            longest_e = []
            for e in effective_es:
                if len(e) > len(longest_e):
                    longest_e = e
            plt.plot(longest_e)
            plt.subplot(224)
            plt.plot(zcrs)
            for ep in endpoints:
                plt.axvline(ep, color='r')
            plt.show()
            plt.waitforbuttonpress()

    def tearDown(self):
        for file in os.listdir(self.DATA_DIR):
            os.remove(os.path.join(self.DATA_DIR, file))
        if os.path.exists(self.DATA_DIR):
            os.removedirs(self.DATA_DIR)
        if os.path.exists(self.LOG):
            os.remove(self.LOG)


if __name__ == '__main__':
    unittest.main()
    #a = PreProcessingTests()
    #a.test_all()