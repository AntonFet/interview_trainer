import sounddevice as sd

class AudioPlayer:

    def play(self, wav, samplerate=22050):

        sd.play(wav, samplerate=samplerate)
        sd.wait()