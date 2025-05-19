from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

# Uloží komplet model a všetky potrebné súbory vrátane preprocessor_config.json
model.save_pretrained("./models/whisper-medium")
processor.save_pretrained("./models/whisper-medium")
