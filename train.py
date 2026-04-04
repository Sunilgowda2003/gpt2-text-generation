import os
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset('text', data_files={'train': 'data.txt'})

# -----------------------------
# Load Tokenizer & Model
# -----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# -----------------------------
# Tokenization
# -----------------------------
def tokenize_function(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # IMPORTANT: add labels for training
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# ✅ APPLY TOKENIZATION (IMPORTANT)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# -----------------------------
# Train Model
# -----------------------------
trainer.train()

# -----------------------------
# Save Model
# -----------------------------
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# -----------------------------
# Text Generation (FIXED)
# -----------------------------
input_text = "Artificial Intelligence"
inputs = tokenizer(input_text, return_tensors="pt")

output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

print("\nGenerated Text:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))