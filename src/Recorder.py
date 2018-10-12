import pyaudio
import wave
import numpy as np


class Recorder:
    """
    A recorder based on PyAudio.
    """

    def __init__(self, chunk=1024, audio_format=pyaudio.paInt16, channels=1, rate=8000):
        """
        The construction function initializes the recorder but does not start the input stream.
        :param chunk: buffer size
        :param audio_format: data format
        :param channels: num of input channel
        :param rate: sample rate
        """
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
        """
        Record a piece of sound.
        :param sec: durance of recording
        :param file_name: save the sound to this file. If None is provided then no file is saved.
        :return: a 1-D numpy array containing the wave
        """
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
