import pandas as pd
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer

# Naèítanie dát
df = pd.read_csv("train.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))

# Naèítanie modelu a tokenizeru
model_name = "openai/whisper-small"  # Môeš poui aj "tiny" pre menší model
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Predspracovanie dát
def preprocess(batch):
    audio = batch["audio_file"]
    inputs = processor(audio["array"], sampling_rate=16000, return_tensors="pt")
    labels = processor.tokenizer(batch["text"], return_tensors="pt").input_ids
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels[0]
    return batch

dataset = dataset.map(preprocess)

# Tréningové parametre
training_args = TrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=5,
    save_steps=100,
    logging_steps=50,
    fp16=True,  # Ak máš GPU s FP16 podporou
    evaluation_strategy="no",
)

# Spustenie tréningu
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()

# Uloenie modelu
trainer.save_model("./whisper-finetuned")

