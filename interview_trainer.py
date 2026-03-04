import pyaudio
import wave
import time
import math
import struct
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import ollama
from TTS.api import TTS

# --- Настройки ---
WAKE_WORD = "start"
CHUNK = 1024
RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
THRESHOLD = 300
SILENCE_LIMIT = 2.0

# --- Инициализация ---
# 1. Модель распознавания речи (Whisper)
print("Загрузка модели Whisper...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# 2. Модель синтеза речи Coqui TTS (многоголосая VCTK)
print("Загрузка Coqui TTS (VCTK)...")
#tts = TTS(
#    model_name="tts_models/en/vctk/vits",
#    progress_bar=False,
#    device="cpu"          # на M2 можно попробовать True, если установлены драйверы
#)
try:
    # Пробуем современный способ с device
    tts = TTS(
        model_name="tts_models/en/vctk/vits",
        progress_bar=False,
#        device="mps"   # или "cpu"
    )
except TypeError:
    # Если не принимает device – создаём без него
    tts = TTS(
        model_name="tts_models/en/vctk/vits",
        progress_bar=False
    )
    print("Используется устройство по умолчанию (скорее всего CPU)")
# Для использования MPS на Mac (если поддерживается) можно указать device="mps":
# tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, device="mps")

# Выбираем голос: p303 — мужской, уверенный, средних лет
# Полный список доступных дикторов можно получить так:
# print(tts.speakers)  # раскомментируйте при необходимости
SELECTED_SPEAKER = "p303"   # при необходимости замените на другой ID

# 3. История диалога для LLM
messages = [
    {"role": "system", "content": """
Ты — технический интервьюер для позиции Senior QA Engineer.
Твоя задача — проводить структурированное собеседование. Задавай только один вопрос за раз.
После каждого ответа кандидата давай краткую обратную связь (на английском) по следующим пунктам:
- Что было хорошо (What was good)
- Что можно улучшить (What could be improved)
- Затем задавай следующий вопрос.

Начни интервью с приветствия и первого вопроса.
Пример: "Hello! Let's start your Senior QA interview. Can you explain the difference between verification and validation?"
    """}
]

# --- Вспомогательные функции ---
def rms(data):
    count = len(data) // 2
    shorts = struct.unpack_from("<" + "h" * count, data)
    sum_squares = sum(sample * sample for sample in shorts)
    return math.sqrt(sum_squares / count) if count > 0 else 0

def record_audio(filename="temp.wav"):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    frames = []
    silence_counter = 0
    print("\n🎤 Слушаю... (говорите)")

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        vol = rms(data)

        if vol < THRESHOLD:
            silence_counter += 1
        else:
            silence_counter = 0

        if silence_counter * (CHUNK / RATE) > SILENCE_LIMIT:
            break

    print("✅ Запись завершена.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def transcribe_audio(filename):
    segments, _ = whisper_model.transcribe(filename, language="en")
    return " ".join([segment.text for segment in segments])

def speak_text(text):
    """Озвучка текста через Coqui TTS (выбранный голос)"""
    print(f"🤖 Интервьюер: {text}")
    try:
        # Генерация речи в numpy-массив (частота 22050 Гц)
        wav = tts.tts(text=text, speaker=SELECTED_SPEAKER)
        # Воспроизведение
        sd.play(wav, samplerate=22050)
        sd.wait()  # дождаться окончания воспроизведения
    except Exception as e:
        print(f"⚠️ Ошибка синтеза речи: {e}")

def ask_llm(user_response):
    messages.append({"role": "user", "content": user_response})
    response = ollama.chat(model='mistral:7b-instruct', messages=messages)
    ai_message = response['message']['content']
    messages.append({"role": "assistant", "content": ai_message})
    return ai_message

# --- Основной цикл программы ---
if __name__ == "__main__":
    print("🚀 Добро пожаловать в AI-тренажер для QA-интервью!")
    print("Говорите 'start', чтобы начать, или просто ждите первого вопроса.")

    # Начальное приветствие от ИИ
    first_question = ask_llm("Let's begin the interview.")
    speak_text(first_question)

    while True:
        audio_file = record_audio("response.wav")
        user_text = transcribe_audio(audio_file)
        if not user_text:
            print("😶 Вас не слышно или ничего не сказано. Попробуйте еще раз.")
            continue
        print(f"👤 Вы сказали: {user_text}")

        if "exit interview" in user_text.lower() or "quit" in user_text.lower():
            speak_text("Thank you for the interview. Good luck with your job search!")
            break

        ai_response = ask_llm(user_text)
        speak_text(ai_response)
        time.sleep(1)