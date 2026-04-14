import streamlit as st
import pandas as pd
import altair as alt
import os

# --- Page Config ---
st.set_page_config(page_title="NLI Diagnostics", layout="wide")
st.title("Partial Input Dependence in Natural Language Inference: Diagnostic Dashboard")

st.markdown("""
**Welcome to the Diagnostic Dashboard.** This tool visualizes how much Natural Language Inference (NLI) models rely on biased shortcuts.
* **Hyp-Only / Prem-Only:** We test the model by hiding the Premise or the Hypothesis to see if it can "guess" the answer without full context.
* **PIRI_hyp (Partial Input Reliance Index):** Measures the drop in accuracy when the premise is removed. A lower PIRI means the model is dangerously good at guessing using *only* the hypothesis.
""")

# --- Data Loading ---
@st.cache_data
def load_summary():
    file_path = "results/summary.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_summary()

if df is None:
    st.error("Could not find 'results/summary.csv'. Please run `python run_diagnostics.py` first.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Explore Results")
models_avail = df['Model'].unique()
datasets_avail = df['Dataset'].unique()

sel_model = st.sidebar.selectbox("Model", models_avail, help="Select the neural architecture to analyze.")
sel_dataset = st.sidebar.selectbox("Dataset", datasets_avail, help="Select the benchmark dataset.")

# --- Main Dashboard ---
filtered_df = df[(df['Model'] == sel_model) & (df['Dataset'] == sel_dataset)]

if filtered_df.empty:
    st.warning("No data available for this Model + Dataset combination.")
else:
    row = filtered_df.iloc[0]
    
    st.subheader(f"Metrics: {sel_model.upper()} on {sel_dataset.upper()}")
    
    # 1. Top Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Full Acc", f"{row['Full_Acc']*100:.1f}%", help="Accuracy when both Premise and Hypothesis are provided.")
    m2.metric("Hyp-Only Acc", f"{row['Hyp_Acc']*100:.1f}%", help="Accuracy when the Premise is hidden.")
    m3.metric("Prem-Only Acc", f"{row['Prem_Acc']*100:.1f}%", help="Accuracy when the Hypothesis is hidden.")
    m4.metric("PIRI_hyp", f"{row['PIRI_hyp']:.3f}", help="Formula: (Full_Acc - Hyp_Acc) / Full_Acc")

    st.markdown("---")
    
    # 2. Charts & Global Summary
    col_chart, col_img = st.columns([1, 1])
    
    with col_chart:
        st.write("**Condition Accuracy Dropoff**")
        # Prepare data for Altair chart
        chart_data = pd.DataFrame({
            'Condition': ['1. Full Input', '2. Hyp-Only', '3. Prem-Only'],
            'Accuracy': [row['Full_Acc'], row['Hyp_Acc'], row['Prem_Acc']]
        })
        
        bar_chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Condition:N', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Accuracy:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
            color=alt.Color('Condition:N', legend=None, scale=alt.Scale(range=['#4C72B0', '#DD8452', '#55A868']))
        ).properties(height=300)
        
        st.altair_chart(bar_chart, use_container_width=True)

    with col_img:
        st.write("**Global Vulnerability Map**")
        img_path = "piri_vs_hans_scatter.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Scatter plot not found. Run your plotting script to generate 'piri_vs_hans_scatter.png'.")

# --- NEW: ADVANCED DETAILS EXPANDER ---
    with st.expander("Advanced Details: Hypothesis-Only Bias Breakdown"):
        csv_path = f"results/{sel_dataset}_{sel_model}_results.csv"
        if os.path.exists(csv_path):
            detail_df = pd.read_csv(csv_path)
            
            # Calculate what % of the time it guesses Entailment(0), Neutral(1), Contradiction(2)
            preds = detail_df['pred_hyp_only'].value_counts(normalize=True) * 100
            
            st.write("When deprived of the premise, how often does the model default to each class?")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicts Entailment", f"{preds.get(0, 0):.1f}%")
            c2.metric("Predicts Neutral", f"{preds.get(1, 0):.1f}%")
            c3.metric("Predicts Contradiction", f"{preds.get(2, 0):.1f}%")
            
            st.info("If one class heavily dominates (e.g., >60%), the model has a strong fallback heuristic. Notice how most models default to 'Neutral' when they can't see the premise.")
        else:
            st.write("Detailed logs not found for this combination.")

# --- Live Interactive Probe ---
st.markdown("---")
st.subheader("Live Inference Probe")
st.write("Type a premise and hypothesis to see how the selected model behaves when deprived of context.")

premise_text = st.text_area("Premise", "A man in a blue shirt is standing in front of a garage playing a guitar.")
hypothesis_text = st.text_input("Hypothesis", "A man is playing an instrument.")

if st.button("Run Live Inference"):
    with st.spinner(f"Loading {sel_model.upper()} into memory and predicting..."):
        from models import get_model
        
        @st.cache_resource
        def load_cached_model(m_key):
            return get_model(m_key)
        
        model = load_cached_model(sel_model)
        
        # --- NEW: Interpretability Probe for RoBERTa ---
        if "roberta" in sel_model.lower():
            st.markdown("### Interpretability Probe (RoBERTa)")
            st.info("Exploratory heuristic: Highlighting tokens based on the average attention from the [CLS] token in the final neural layer.")
            
            p_full, tokens, attentions = model.predict_with_attention(premise_text, hypothesis_text)
            
            # Normalize attention to scale opacity cleanly (max token gets ~80% opacity)
            max_att = max(attentions) if max(attentions) > 0 else 1
            
            html = "<div style='line-height: 2; font-size: 1.1em; padding: 10px; background-color: #f0f2f6; border-radius: 8px;'>"
            for tok, att in zip(tokens, attentions):
                # Clean up RoBERTa's subword artifacts and skip special structural tokens
                clean_tok = tok.replace('Ġ', ' ').replace('</s>', '').replace('<s>', '')
                if not clean_tok.strip(): 
                    continue
                
                intensity = (att / max_att) * 0.8 
                html += f"<span style='background-color: rgba(76, 114, 176, {intensity}); padding: 2px 4px; border-radius: 4px; margin-right: 2px;'>{clean_tok}</span>"
            html += "</div><br>"
            
            st.markdown(html, unsafe_allow_html=True)
        else:
            p_full = model.predict(premise_text, hypothesis_text)
        # -----------------------------------------------
        
        # Run remaining conditions
        p_hyp = model.predict("", hypothesis_text)
        p_prem = model.predict(premise_text, "")
        
        label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        
        r1, r2, r3 = st.columns(3)
        r1.success(f"**Full Input:**\n\n{label_map.get(p_full, 'Error')}")
        r2.warning(f"**Hyp-Only Input:**\n\n{label_map.get(p_hyp, 'Error')}")
        r3.error(f"**Prem-Only Input:**\n\n{label_map.get(p_prem, 'Error')}")