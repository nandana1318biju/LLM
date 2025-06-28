import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

# Select 3–5 test samples
def sample_test_data(prepared_data, num_samples=5):
    indices = random.sample(range(len(prepared_data["prompts"])), num_samples)
    samples = [{
        "prompt": prepared_data["prompts"][i],
        "actual_answer": prepared_data["answers"][i]
    } for i in indices]
    return samples

# Load model/tokenizer without pipeline
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# Generate answer using manual tokenization and model inference
def generate_answer(prompt, tokenizer, model, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

# ==== MODELS TO COMPARE ====
model_dict = {
    "GPT-2 (Foundation)": "gpt2",
    "BioGPT": "microsoft/BioGPT",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
}

# === Load prepared data from Step 1 (assuming in same file or import) ===
import pickle

# If you're loading from Step 1 result saved file
# with open("prepared_data.pkl", "rb") as f:
#     prepared_data = pickle.load(f)

# Otherwise (if running immediately after Step 1), just import prepared_data:
from data_preparation import prepared_data

# === Sample test questions ===
test_samples = sample_test_data(prepared_data, num_samples=5)

# === Output Storage ===
results = []

# === Loop through models ===
for model_name, model_path in model_dict.items():
    print(f"\n🔍 Running model: {model_name}")
    tokenizer, model = load_model(model_path)

    for sample in test_samples:
        generated = generate_answer(sample["prompt"], tokenizer, model)
        results.append({
            "model": model_name,
            "prompt": sample["prompt"],
            "actual_answer": sample["actual_answer"],
            "generated_answer": generated
        })
        print(f"\nModel: {model_name}")
        print("Prompt:", sample["prompt"])
        print("Actual:", sample["actual_answer"])
        print("Generated:", generated)

# Optionally save to CSV or JSON
import pandas as pd
df_results = pd.DataFrame(results)
df_results.to_csv("qa_model_outputs.csv", index=False)
print("\n✅ Saved outputs to 'qa_model_outputs.csv'")
