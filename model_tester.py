import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_single_inference():
    # Swapped to a public, stable RoBERTa NLI model
    model_name = "cross-encoder/nli-roberta-base"
    
    print(f"Loading tokenizer and model: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    model.eval()

    premise = "A dog is running in the park."
    hypothesis = "An animal is moving outside."
    
    print("\nTokenizing inputs...")
    inputs = tokenizer(
        premise, 
        hypothesis, 
        return_tensors="pt",
        truncation=True
    )
    
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    # Mapping for this specific model: 0=contradiction, 1=entailment, 2=neutral
    label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
    predicted_label = label_map.get(predicted_class_id, f"UNKNOWN_ID_{predicted_class_id}")
    
    print("\n--- Results ---")
    print(f"Premise:    {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Raw Logits: {logits.tolist()[0]}")
    print(f"Prediction: {predicted_label} (Class ID: {predicted_class_id})")

if __name__ == "__main__":
    test_single_inference()