import streamlit as st
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# Load model outputs
@st.cache_data
def load_data():
    return pd.read_csv("qa_model_outputs.csv")

df = load_data()
model_names = df['model'].unique()

# Sidebar for model selection
selected_model = st.selectbox("Select a Model", model_names)

# Filter data by model
filtered_df = df[df['model'] == selected_model]

st.title("🧠 Medical QA Model Comparison")
st.subheader(f"Model: {selected_model}")

# Display outputs
for i, row in filtered_df.iterrows():
    st.markdown(f"**Q:** {row['prompt']}")
    st.markdown(f"**Actual Answer:** {row['actual_answer']}")
    st.markdown(f"**Generated Answer:** {row['generated_answer']}")
    st.markdown("---")

# === Evaluation Metrics ===

def compute_scores(df):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    model_scores = {}

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        bleu_scores = []
        rouge_l_scores = []

        for _, row in model_df.iterrows():
            reference = row['actual_answer']
            candidate = row['generated_answer']
            bleu = sentence_bleu(
                [reference.split()],
                candidate.split(),
                smoothing_function=SmoothingFunction().method1
            )
            rouge_l = scorer.score(reference, candidate)['rougeL'].fmeasure
            bleu_scores.append(bleu)
            rouge_l_scores.append(rouge_l)

        model_scores[model] = {
            'BLEU': sum(bleu_scores) / len(bleu_scores),
            'ROUGE-L': sum(rouge_l_scores) / len(rouge_l_scores)
        }

    return model_scores

# Compute and show bar chart
model_scores = compute_scores(df)

# Plot
st.subheader("📊 Model Evaluation Scores")
score_df = pd.DataFrame(model_scores).T.reset_index().rename(columns={"index": "Model"})

st.dataframe(score_df.style.format({"BLEU": "{:.4f}", "ROUGE-L": "{:.4f}"}))

st.bar_chart(score_df.set_index("Model"))

