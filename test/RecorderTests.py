import unittest
import sys
import os
lib_path = os.path.abspath(os.path.join(sys.path[0], '..'))
sys.path.append(lib_path)
import matplotlib.pyplot as plt
from src.Recorder import Recorder
import numpy as np
from multiprocessing import Process
from queue import Queue


def test_stream_service(queue, recorder):
    plt.ion()
    plt.figure(1)
    plt.show()
    TEST_FRAME_NUM = 100
    PRE_FRAME_NUM = 20
    interval = recorder.CHUNK / recorder.RATE
    t = np.linspace(0, (TEST_FRAME_NUM + 1) * interval, TEST_FRAME_NUM)
    t_now = 0
    state = 0
    rec = []
    recording = False
    THRESHOLD = 5e10
    noise_frames = []
    for i in range(10):
        noise = queue.get(True).astype(np.float32)
        noise_frames.append(noise)
        print(noise)
    noise_frames = np.concatenate(noise_frames)
    energy = np.sum(np.square(noise_frames))
    avg = np.average(energy)
    variance = np.var(energy)
    print(avg)
    THRESHOLD = avg + 5 * np.sqrt(variance)
    print("THRESHOLD =", THRESHOLD)
    leading_frame = Queue(PRE_FRAME_NUM)

    for i in range(TEST_FRAME_NUM):
        wav = queue.get(True).astype(np.float32)
        energy = np.sum(wav * wav)
        queue_size = queue.qsize()
        t_now += interval
        if energy > THRESHOLD:
            if state < 2:
                state += 1
            else:
                recording = True
                state = 20
        else:
            if state > 0:
                if recording:
                    state -= 1
            else:
                if recording:
                    plt.clf()
                    wav_full = np.concatenate(rec)
                    leading_frame_full = []
                    while not leading_frame.empty():
                        leading_frame_temp = leading_frame.get(True).astype(np.float32)
                        leading_frame_full.append(leading_frame_temp)
                    leading_frame_full = np.concatenate(leading_frame_full)
                    wav_full = np.concatenate((leading_frame_full, wav_full))
                    plt.subplot(121)
                    plt.plot(wav_full)
                    plt.subplot(122)
                    plt.plot(leading_frame_full)
                    plt.draw()
                    plt.pause(0.001)
                rec.clear()
                recording = False
        if recording:
            rec.append(wav)
        else:
            if leading_frame.full():
                leading_frame.get(True)
            leading_frame.put(wav)
    print("EXIT")
    recorder.stop_streaming()


class RecorderTests(unittest.TestCase):

    def setUp(self):
        self.recorder = Recorder()
        self.test_stream_proc = Process(target = test_stream_service,
                                        args=(self.recorder.stream_queue, self.recorder))

    def test_rec_one_shot(self):
        for i in range(3):
            wav = self.recorder.rec_one_shot(sec=1)
            self.assertFalse(np.abs(wav).max() == 0)
            plt.plot(wav)
            plt.title('Test ' + str(i))
            plt.waitforbuttonpress()

    def test_stream(self):
        self.recorder.start_streaming()
        self.test_stream_proc.run()
        print("Stoped")


if __name__ == '__main__':
    unittest.main()
