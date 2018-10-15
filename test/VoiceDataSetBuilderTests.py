import unittest
import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.VoiceDataSetBuilder import *


class TestVoiceDataSetBuilder(unittest.TestCase):

    def setUp(self):
        self.LOG = str(datetime.now()).replace(' ', '-') + '.txt'
        self.DST_PATH = './TEST_DATA'
        self.builder = VoiceDataSetBuilder(self.DST_PATH, log_file=self.LOG)

    def test_generate_name(self):
        label1 = str(datetime.now()).replace(' ', '-') + '.wav'
        gen_name, rnd = self.builder.generate_name(label1).split('_')
        print(gen_name)
        self.assertEqual(label1, gen_name)

    def test_build(self):
        self.builder.build()

    def tearDown(self):
        for file in os.listdir(self.DST_PATH):
            os.remove(os.path.join(self.DST_PATH, file))
        if os.path.exists(self.DST_PATH):
            os.removedirs(self.DST_PATH)
        if os.path.exists(self.LOG):
            os.remove(self.LOG)



if __name__ == '__main__':
    unittest.main()
