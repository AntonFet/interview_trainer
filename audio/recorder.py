import logging
import wave
import pyaudio

from utils.audio_utils import rms
from config import *

logger = logging.getLogger(__name__)


class AudioRecorder:

    def record(self, filename="temp.wav"):

        logger.info("Starting audio recording")

        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        frames = []
        silence_counter = 0

        while True:

            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            vol = rms(data)

            if vol < THRESHOLD:
                silence_counter += 1
            else:
                silence_counter = 0

            if silence_counter * (CHUNK / RATE) > SILENCE_LIMIT:
                logger.info("Silence detected, stopping recording")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()

        logger.info("Audio saved to %s", filename)

        return filename