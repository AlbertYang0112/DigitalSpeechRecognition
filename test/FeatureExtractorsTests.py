import unittest
import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.FeatureExtractors import *
from src.PreProcessing import *
import matplotlib.pyplot as plt

class FeatureExtractorsTests(unittest.TestCase):

    def setUp(self):
        self.WAVE_NUM = 10
        self.WAVE_DURANCE = 1000
        self.wave_list = np.random.randn(self.WAVE_DURANCE, self.WAVE_NUM)
        self.base = PreProcessing(100, 10)
        self.frames = np.array([self.base.Enframe(wave) for wave in self.wave_list])

    def test_enhance_frame(self):
        for i in range(self.WAVE_NUM):
            self.assertTrue((
                FeatureExtractors.enhance_frame(
                    wav_data=self.wave_list[:, i],
                    frame_size=100,
                    overlap=10,
                    windowing_method='Hamming'
                ) ==
                self.base.Enframe(self.wave_list[:, i])
            ).all())

    def test_zero_crossing_rate(self):
        for i in range(self.WAVE_NUM):
           self.assertTrue((
               FeatureExtractors.zero_crossing_rate(
                   frames=self.frames
               ) ==
               self.base.ZCR(self.frames)
           ).any())

    def test_energy(self):
        for i in range(self.WAVE_NUM):
            self.assertTrue((
                FeatureExtractors.energy(self.frames) ==
                self.base.energy(self.frames)
            ).any())

if __name__ == '__main__':
    unittest.main()
