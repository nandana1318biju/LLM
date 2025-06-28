import pandas as pd
from transformers import AutoTokenizer

# === Load Dataset ===
df = pd.read_csv("medquad.csv")

# === Keep only required rows ===
df = df[['question', 'answer']].dropna().head(4000)

# === Prepare Prompt Format ===
df['prompt'] = df['question'].apply(lambda x: f"Q: {x}\nA:")

# === Select Tokenizer (change as needed for BioGPT / PubMedBERT) ===
model_name = "gpt2"  # Change to 'microsoft/BioGPT' or 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 has no pad_token by default, so we add it manually
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Tokenize and Pad/Truncate ===
max_length = 128

tokenized_inputs = tokenizer(
    list(df['prompt']),
    max_length=max_length,
    truncation=True,
    padding='max_length',
    return_tensors="pt"
)

tokenized_outputs = tokenizer(
    list(df['answer']),
    max_length=max_length,
    truncation=True,
    padding='max_length',
    return_tensors="pt"
)

# === Save tokenized tensors and original prompts for future use ===
prepared_data = {
    "inputs": tokenized_inputs,
    "labels": tokenized_outputs,
    "prompts": df['prompt'].tolist(),
    "answers": df['answer'].tolist()
}

print("✅ Data preparation complete.")
