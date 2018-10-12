import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
from src.VoiceDataSetBuilder import VoiceDataSetBuilder
builder = VoiceDataSetBuilder('../DataSet', rate=44100, log_file='../DataSet/DataList.txt')
builder.build()
