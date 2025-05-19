import os, sys
import sounddevice as sd  # type: ignore
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import pyttsx3
from datetime import datetime
import edge_tts
import asyncio
from playsound import playsound





text_file_path = "output.txt"  # Súbor na uloženie rozpoznaného textu

# 🔀 Prepínače modelov
use_finetuned = False   # True = doladený model, False = predtrénovaný
pretrained_model_size = "medium"  # "small" alebo "medium"

# 📚 Cesty k modelom
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = "whisper-finetuned" if use_finetuned else f"whisper-{pretrained_model_size}"
model_path = os.path.normpath(os.path.join(base_dir, "models", model_dir))

# Načítanie procesora a modelu
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# Odstránenie forced_decoder_ids (zabránime chybe)
if hasattr(model.config, "forced_decoder_ids"):
    model.config.forced_decoder_ids = None

# Inicializácia pipeline bez použitia ffmpeg
asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    framework="pt"  # Používame PyTorch
)

# 🎙️ Nahrávanie zvuku
samplerate = 16000
duration = 10  # Dĺžka nahrávky v sekundách

print("🎙️ Nahrávam cez mikrofón...")
try:
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
except Exception as e:
    print(f"Chyba pri nahrávaní zvuku: {e}")
    exit(1)

# Prevod na 1D pole
audio_data = np.squeeze(audio)

print("✅ Zvuk nahratý, rozpoznávam reč...")

result = asr(
    audio_data,
    chunk_length_s=10,
    generate_kwargs={"max_new_tokens": 256, "language": "sk"},  # Nastavenie jazyka na slovenčinu
    return_timestamps="none"
)

recognized_text = result["text"]

# 🗣️ Výstup rozpoznaného textu
print("\n🗣️ Rozpoznané:")
print(recognized_text)

# Kontrola na kľúčové slovo "stačí"
if "stačí" in recognized_text.lower():
    print("🛑 Zaznamenané 'stačí'. Ukončujem nahrávanie.")

# Uloženie výsledku do textového súboru
with open(text_file_path, "a", encoding="utf-8") as text_file:
    text_file.write(f"{datetime.now()}: {recognized_text}\n")







text_file_path = "output.txt"

# 1. Načítaj posledný riadok zo súboru
with open(text_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    if lines:
        posledny = lines[-1].strip().split(":", 1)[-1].strip()
    else:
        posledny = "Súbor je prázdny."

print(f"🗣️ Čítam: {posledny}")

# 2. Slovenský hlas cez edge-tts (Jakub, alebo zmeň na Viktoria)
async def tts(text):
    communicate = edge_tts.Communicate(text, "sk-SK-JakubNeural")
    print(f"Text na čítanie: '{posledny}'")
    await communicate.save("output.mp3")

asyncio.run(tts(posledny))

# 3. Prehrať audio
playsound("output.mp3")




print("\n✅ Skript úspešne dokončený.")

sys.exit(0)
