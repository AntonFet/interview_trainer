import logging
from TTS.api import TTS
from config import TTS_MODEL, TTS_SPEAKER

logger = logging.getLogger(__name__)


class TTSService:

    def __init__(self):

        logger.info("Loading Coqui TTS model")

        try:
            self.tts = TTS(
                model_name=TTS_MODEL,
                progress_bar=False
            )
        except TypeError:

            logger.warning("Device parameter not supported, fallback to default")

            self.tts = TTS(model_name=TTS_MODEL)

        self.speaker = TTS_SPEAKER

        logger.info("TTS model loaded, speaker=%s", self.speaker)

    def synthesize(self, text):

        logger.info("Synthesizing speech")

        try:
            wav = self.tts.tts(
                text=text,
                speaker=self.speaker
            )

            return wav

        except Exception as e:

            logger.error("TTS synthesis failed: %s", e)

            raise