import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from interpretability import NLIExplainer

@st.cache_resource
def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure output_attentions=True for the Attention Heatmap feature
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

def main():
    st.title("NLI-PIRI Interpretability Dashboard")

    st.markdown("""
    > **Caveat:** Attention heatmaps and gradient-based attributions are illustrative tools to understand model focus. They do not represent strict causal logic or guarantee the model's true reasoning process. Interpret with caution.
    """)

    st.sidebar.header("Configuration")
    # Add supported models here aligning with config.py
    model_choice = st.sidebar.selectbox(
        "Model", 
        ["roberta-base", "microsoft/deberta-base", "distilroberta-base", "distilbert-base-uncased", "facebook/bart-large-mnli"]
    )

    model, tokenizer, device = load_model_and_tokenizer(model_choice)

    st.header("Input")
    premise = st.text_area("Premise", "A man is playing a guitar on stage.")
    hypothesis = st.text_input("Hypothesis", "A man is making music.")

    if st.button("Analyze"):
        if not premise or not hypothesis:
            st.warning("Please enter both a premise and a hypothesis.")
            return

        st.header("Prediction")
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            prediction = torch.argmax(logits, dim=-1).item()
        
        # Adjust mapping if your models use a different label order
        classes = ["Entailment", "Neutral", "Contradiction"]
        
        if len(probs) == 2:
            classes = ["Entailment", "Non-Entailment"]

        st.write(f"**Predicted Class:** {classes[prediction]} (Confidence: {probs[prediction]:.4f})")
        
        st.header("Interpretability")
        method = st.radio("Select Attribution Method", ["Attention Heatmap", "Integrated Gradients"])

        if method == "Attention Heatmap":
            st.subheader("Attention Heatmap")
            st.write("Displaying averaged [CLS] token attention over the sequence from the final layer.")
            
            if outputs.attentions:
                # Extract last layer attentions, average across heads, take CLS row (index 0)
                last_layer_attn = outputs.attentions[-1][0].mean(dim=0)[0].cpu().numpy()
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                
                html_string = "<div>"
                for token, score in zip(tokens, last_layer_attn):
                    # Scale intensity for visibility
                    color = f"rgba(255, 0, 0, {score * 5})"
                    html_string += f"<span style='background-color: {color}; padding: 2px; margin: 1px; border-radius: 3px; display: inline-block;'>{token}</span>"
                html_string += "</div>"
                st.markdown(html_string, unsafe_allow_html=True)
            else:
                st.warning("Model does not output attentions. Verify configuration.")

        elif method == "Integrated Gradients":
            st.subheader("Integrated Gradients Attribution")
            explainer = NLIExplainer(model, tokenizer)
            
            target_class = st.selectbox(
                "Target Class for Attribution", 
                range(len(classes)), 
                index=prediction, 
                format_func=lambda x: classes[x]
            )
            
            with st.spinner("Computing Integrated Gradients..."):
                try:
                    tokens, scores = explainer.explain_integrated_gradients(premise, hypothesis, target_class)
                    
                    html_string = "<div>"
                    for token, score in zip(tokens, scores):
                        if score > 0:
                            color = f"rgba(255, 0, 0, {min(score * 3, 1.0)})"
                        else:
                            color = f"rgba(0, 0, 255, {min(abs(score) * 3, 1.0)})"
                        html_string += f"<span style='background-color: {color}; padding: 2px; margin: 1px; border-radius: 3px; display: inline-block;'>{token}</span>"
                    html_string += "</div>"
                    
                    st.markdown(html_string, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Integrated Gradients computation failed. Note: IG requires specific embedding layer extraction which may not be mapped for this exact HF architecture. Error: {e}")

if __name__ == "__main__":
    main()