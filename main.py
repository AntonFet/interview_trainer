import time
import logging

from audio.recorder import AudioRecorder
from audio.player import AudioPlayer
from stt.whisper_service import WhisperService
from tts.tts_service import TTSService
from llm.interview_agent import InterviewAgent

logger = logging.getLogger(__name__)


def main():

    recorder = AudioRecorder()
    player = AudioPlayer()

    stt = WhisperService()
    tts = TTSService()
    agent = InterviewAgent()

    stage = "QUESTION"

    print("🚀 QA Interview AI started")

    while True:

        # ----------------
        # 1. AI задает вопрос
        # ----------------
        if stage == "QUESTION":

            question = agent.generate_question()

            print("AI:", question)

            wav = tts.synthesize(question)

            player.play(wav)

            stage = "WAIT_USER"

        # ----------------
        # 2. Ждем ответ пользователя
        # ----------------
        elif stage == "WAIT_USER":

            audio_file = recorder.record("response.wav")

            user_text = stt.transcribe(audio_file)

            if not user_text.strip():

                print("Didn't hear anything")

                continue

            print("User:", user_text)

            stage = "FEEDBACK"

        # ----------------
        # 3. AI дает feedback
        # ----------------
        elif stage == "FEEDBACK":

            feedback = agent.generate_feedback(user_text)

            print("AI:", feedback)

            wav = tts.synthesize(feedback)

            player.play(wav)

            stage = "QUESTION"

        time.sleep(1)


if __name__ == "__main__":
    main()