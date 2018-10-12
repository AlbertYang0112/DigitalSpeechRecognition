import pyaudio
import wave
import numpy as np


class Recorder:

    def __init__(self, chunk=1024, audio_format=pyaudio.paInt16, channels=1, rate=8000):
        self.CHUNK = chunk
        self.FORMAT = audio_format
        self.CHANNELS = channels
        self.RATE = rate
        self.recorder = pyaudio.PyAudio()
        self.stream = self.recorder.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        self.stream.stop_stream()

    def rec_one_shot(self, sec, file_name=None):
        self.stream.start_stream()
        frames = []
        for i in range(int(self.RATE / self.CHUNK * sec)):
            data = self.stream.read(self.CHUNK)
            data = np.fromstring(data, dtype=np.int16)
            frames.append(data)
        self.stream.stop_stream()
        if file_name is not None:
            with wave.open(file_name, 'wb') as wav_file:
                wav_file.setnchannels(self.CHANNELS)
                wav_file.setsampwidth(self.recorder.get_sample_size(self.FORMAT))
                wav_file.setframerate(self.RATE)
                wav_file.writeframes(b''.join(frames))
        frame = np.concatenate(frames, 0)
        return frame

    def save_wav(self, wav, file_name):
        with wave.open(file_name, 'wb') as wav_file:
            wav_file.setnchannels(self.CHANNELS)
            wav_file.setsampwidth(self.recorder.get_sample_size(self.FORMAT))
            wav_file.setframerate(self.RATE)
            wav_file.writeframes(b''.join(wav))
