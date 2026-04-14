import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODELS

class NLIModel:
    def __init__(self, model_name="cross-encoder/nli-roberta-base"):
        print(f"Loading model and tokenizer: '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # --- FIX: Force 'eager' attention so we can extract the weights later ---
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            attn_implementation="eager"
        )
        self.model.eval()
        
        # Cross-encoders (and our fine-tuned local model) output: 0=contradiction, 1=entailment, 2=neutral
        if "cross-encoder" in model_name or "piri_model" in model_name:
            self._model_to_snli_map = {0: 2, 1: 0, 2: 1}
        # Typeform DistilBERT outputs: 0=entailment, 1=neutral, 2=contradiction
        elif "typeform" in model_name:
            self._model_to_snli_map = {0: 0, 1: 1, 2: 2}
        else:
            self._model_to_snli_map = {0: 0, 1: 1, 2: 2}

    def predict(self, premise, hypothesis):
        """
        Runs inference on a premise-hypothesis pair and returns the SNLI-formatted prediction.
        """
        inputs = self.tokenizer(
            premise, 
            hypothesis, 
            return_tensors="pt", 
            truncation=True
        )
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        pred_id = torch.argmax(logits, dim=-1).item()
        
        # Return the standardized label
        return self._model_to_snli_map[pred_id]
    # Add this inside the NLIModel class
    def predict_with_attention(self, premise, hypothesis):
        """Exploratory heuristic: returns mapped prediction, tokens, and CLS attention weights."""
        import torch
        # Temporarily tell the model to return attention weights
        self.model.config.output_attentions = True
        
        encodings = self.tokenizer(premise, hypothesis, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            pred = torch.argmax(outputs.logits, dim=1).item()
            mapped_pred = self._model_to_snli_map.get(pred, pred)
            
            # Heuristic: Take the last layer's attention, average across all attention heads
            # Shape of outputs.attentions[-1]: (batch_size, num_heads, seq_len, seq_len)
            last_layer_attn = outputs.attentions[-1]
            avg_heads_attn = last_layer_attn.mean(dim=1).squeeze(0)
            
            # Extract attention originating from the [CLS] token (index 0) to all other tokens
            cls_attention = avg_heads_attn[0].cpu().tolist()
            
        # Get actual string tokens for rendering
        tokens = self.tokenizer.convert_ids_to_tokens(encodings.input_ids[0])
        
        # Reset config to keep standard runs fast
        self.model.config.output_attentions = False 
        
        return mapped_pred, tokens, cls_attention

def get_model(model_key):
    """Router function to initialize a model based on string key."""
    model_key = model_key.lower()
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODELS.keys())}")
        
    model_path = MODELS[model_key]
    return NLIModel(model_name=model_path)

if __name__ == "__main__":
    # Self-test with a newer generation DeBERTa model
    test_model_name = "cross-encoder/nli-deberta-v3-base"
    tester = NLIModel(model_name=test_model_name)
    
    p = "A dog is running in the park."
    h = "An animal is moving outside."
    
    pred = tester.predict(p, h)
    print(f"\nTest prediction: {pred} (Expected: 0 for entailment)")