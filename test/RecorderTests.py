import unittest
import matplotlib.pyplot as plt
from src.Recorder import Recorder
import numpy as np


class RecorderTests(unittest.TestCase):

    def setUp(self):
        self.recorder = Recorder()

    def test_rec_one_shot(self):
        for i in range(3):
            wav = self.recorder.rec_one_shot(sec=1)
            self.assertFalse(np.abs(wav).max() == 0)
            plt.plot(wav)
            plt.title('Test ' + str(i))
            plt.show()


if __name__ == '__main__':
    unittest.main()
