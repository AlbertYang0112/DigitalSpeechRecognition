import unittest
import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.VoiceDataSetBuilder import *


if __name__ == '__main__':
    voicedatasetbuilder = VoiceDataSetBuilder(dst_path='../DataSet/syk', log_file='../DataSet/DataList_syk.txt')

    voicedatasetbuilder.build()

