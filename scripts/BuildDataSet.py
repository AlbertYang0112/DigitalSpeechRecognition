from src.VoiceDataSetBuilder import VoiceDataSetBuilder
builder = VoiceDataSetBuilder('../DataSet', rate=44100, log_file='../DataSet/DataList.txt')
builder.build()
