import logging
from faster_whisper import WhisperModel
from config import WHISPER_MODEL

logger = logging.getLogger(__name__)


class WhisperService:

    def __init__(self):

        logger.info("Loading Whisper model...")

        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

        logger.info("Whisper model loaded")

    def transcribe(self, filename):

        logger.info("Starting transcription: %s", filename)

        segments, _ = self.model.transcribe(
            filename,
            language="en"
        )

        text = " ".join(segment.text for segment in segments)

        logger.info("Transcription result: %s", text)

        return text