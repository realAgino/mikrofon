import os
import sounddevice as sd # type: ignore
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

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
audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
sd.wait()

# Prevod na 1D pole
audio_data = np.squeeze(audio)

print("✅ Zvuk nahratý, rozpoznávam reč...")

# Rozpoznanie reči priamo z numpy array bez súborov
#result = asr(audio_data,chunk_length_s=10, max_new_tokens=256)
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
    text_file.write(recognized_text + "\n")

# 📄 Prečítanie obsahu súboru
print("\n📄 Obsah súboru:")
def new_func(text_file):
    print(text_file.read())

with open(text_file_path, "r", encoding="utf-8") as text_file:
    new_func(text_file)

with open(text_file_path, "r", encoding="utf-8") as text_file:
    new_func(text_file)

print("\n✅ Skript úspešne dokončený.")
import sys
sys.exit(0)
