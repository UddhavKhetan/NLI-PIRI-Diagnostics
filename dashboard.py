import streamlit as st
import pandas as pd
import os
import torch
from models import get_model
from config import MODELS, DATASETS

st.set_page_config(page_title="NLI Diagnostics", layout="wide")

st.title("🔍 NLI Partial Input Reliance Index (PIRI)")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Select Model", list(MODELS.keys()))
selected_dataset = st.sidebar.selectbox("Select Dataset", list(DATASETS.keys()))
selected_ablation = st.sidebar.selectbox("Ablation Strategy", ["empty", "mask", "neutral", "random"])

# --- Summary Metrics Section ---
st.header("📊 Evaluation Metrics")
csv_path = f"results/{selected_dataset}_{selected_model}_{selected_ablation}_results.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    # Calculate basic stats from the first seed for display
    seed_0 = df['seed'].unique()[0]
    df_disp = df[df['seed'] == seed_0]
    
    acc_full = (df_disp['pred_full'] == df_disp['gold_label']).mean()
    acc_hyp = (df_disp['pred_hyp_only'] == df_disp['gold_label']).mean()
    piri = 1 - (acc_hyp / acc_full) if acc_full > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Full Input Accuracy", f"{acc_full*100:.1f}%")
    col2.metric("Hypothesis-Only Acc", f"{acc_hyp*100:.1f}%")
    col3.metric("PIRI Score", f"{piri:.3f}")
    col4.metric("Total Samples", len(df_disp))
    
    st.markdown("### Advanced Probabilistic Metrics (Single Seed)")
    # We display the raw columns if advanced metrics aren't pre-calculated in a summary file
    st.dataframe(df_disp[['premise', 'hypothesis', 'gold_label', 'pred_full', 'pred_hyp_only']].head(10), use_container_width=True)
else:
    st.warning(f"No data found for {selected_model} on {selected_dataset} with '{selected_ablation}' ablation. Run the pipeline first.")

# --- Live Interpretability Probe ---
st.markdown("---")
st.header("🔬 Live Interpretability Probe")

premise_input = st.text_input("Premise", "A man is playing an instrument.")
hypothesis_input = st.text_input("Hypothesis", "A human is making music.")

if st.button("Run Live Inference"):
    with st.spinner(f"Loading {selected_model}..."):
        model_instance = get_model(selected_model)
        
        # 1. Full Input
        pred_full, probs_full = model_instance.predict(premise_input, hypothesis_input)
        
        # 2. Hypothesis-Only
        if selected_ablation == "empty":
            ab_str = ""
        elif selected_ablation == "neutral":
            ab_str = "The entity is present."
        elif selected_ablation == "mask" and hasattr(model_instance.tokenizer, 'mask_token'):
            ab_str = " ".join([model_instance.tokenizer.mask_token] * 5)
        else:
            ab_str = "apple guitar window abstract train"
            
        pred_hyp, probs_hyp = model_instance.predict(ab_str, hypothesis_input)
        pred_prem, probs_prem = model_instance.predict(premise_input, ab_str)
        
        label_map_rev = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Full Input:** {label_map_rev.get(pred_full, 'Unknown')}")
        with col2:
            st.warning(f"**Hypothesis-Only:** {label_map_rev.get(pred_hyp, 'Unknown')}")
        with col3:
            st.error(f"**Premise-Only:** {label_map_rev.get(pred_prem, 'Unknown')}")

        # Attention Mapping (BERT-family only)
        if "flan" in selected_model or "bart" in selected_model:
            st.info(f"Attention extraction is disabled for generative/encoder-decoder LLMs ({selected_model}).")
        else:
            try:
                # Assuming predict_with_attention exists for BERT models
                _, tokens, attn = model_instance.predict_with_attention(premise_input, hypothesis_input)
                
                html_str = "<div style='font-size: 18px; line-height: 1.5;'>"
                for word, score in zip(tokens, attn):
                    # Filter subword hashes for display
                    word_clean = word.replace('Ġ', '').replace('##', '')
                    if word_clean in ['<s>', '</s>', '[CLS]', '[SEP]', '<pad>']:
                        continue
                    
                    # Normalize score for visual intensity
                    intensity = int(min(score * 255 * 5, 255))
                    bg_color = f"rgba(255, 0, 0, {intensity/255:.2f})"
                    html_str += f"<span style='background-color: {bg_color}; padding: 2px 4px; margin: 2px; border-radius: 3px;'>{word_clean}</span>"
                html_str += "</div>"
                
                st.markdown("### Extracting [CLS] Token Attention")
                st.components.v1.html(html_str, height=100)
            except Exception as e:
                st.error(f"Attention extraction failed: {str(e)}")
