from src import PreProcessing

filename = '../database/0.wav'
test = PreProcessing.PreProcessing(512, 128)
frame, energy, zcr, endpoint = test.process(filename)
test.print_result(filename, frame, energy, zcr, endpoint)
