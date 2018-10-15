import unittest
import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.FileLoader import *
from src.VoiceDataSetBuilder import *

class FileLoaderTests(unittest.TestCase):

    def setUp(self):
        self.DATA_SET_DIR = './TEST'
        self.LOG = './test_lists.txt'
        builder = VoiceDataSetBuilder(self.DATA_SET_DIR, rec_length=0.5, log_file=self.LOG)
        self.wav_list, self.label_list = builder.build()
        self.loader = FileLoader(self.LOG)
        self.NUM = len(self.wav_list)

    def test_read_next(self):
        for i in range(self.NUM):
            wav, label = self.loader.read_next()
            ground_wav, ground_label = self.wav_list[i], self.label_list[i]
            gmax, gmin = np.max(ground_wav), np.min(ground_wav)
            ground_wav = (ground_wav - gmin) / (gmax - gmin)
            ground_wav = np.where(np.abs(ground_wav) < 0.05, 0, ground_wav)
            err = np.max(np.abs(wav-ground_wav))
            self.assertTrue(np.max(np.abs(wav-ground_wav)) < 0.01)
            self.assertEqual(ground_label, label)

    def tearDown(self):
        if os.path.exists(self.DATA_SET_DIR):
            for file in os.listdir(self.DATA_SET_DIR):
                os.remove(os.path.join(self.DATA_SET_DIR, file))
            os.removedirs(self.DATA_SET_DIR)
        if os.path.exists(self.LOG):
            os.remove(self.LOG)


if __name__ == "__main__":
    unittest.main()
