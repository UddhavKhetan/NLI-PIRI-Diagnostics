import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
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
        
        encodings = self.tokenizer(
            premise, hypothesis, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred = torch.argmax(logits, dim=1).item()
            
            # Map prediction to uniform schema
            mapped_pred = self._model_to_snli_map.get(pred, pred)
            
            # Map probabilities to uniform schema (0: Entailment, 1: Neutral, 2: Contradiction)
            mapped_probs = [0.0, 0.0, 0.0]
            for orig_idx, mapped_idx in self._model_to_snli_map.items():
                mapped_probs[mapped_idx] = float(probs[orig_idx])
                
        return mapped_pred, mapped_probs

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


class GenerativeNLIModel:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Token IDs for 'yes' (entailment), 'maybe' (neutral), 'no' (contradiction)
        self.target_ids = [
            self.tokenizer.encode("yes", add_special_tokens=False)[0],
            self.tokenizer.encode("maybe", add_special_tokens=False)[0],
            self.tokenizer.encode("no", add_special_tokens=False)[0]
        ]

    def predict(self, premise, hypothesis):
        # Instruction tuning prompt
        prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Answer 'yes', 'maybe', or 'no'."
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Get logits for the very first generated token
            outputs = self.model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            first_token_logits = outputs.scores[0][0]
            
            # Isolate probabilities for our 3 target words
            target_logits = torch.tensor([first_token_logits[tid] for tid in self.target_ids])
            probs = F.softmax(target_logits, dim=0).cpu().numpy()
            
            pred_idx = torch.argmax(target_logits).item()
            
        # Returns 0 (Entailment), 1 (Neutral), or 2 (Contradiction) along with probabilities
        return pred_idx, [float(probs[0]), float(probs[1]), float(probs[2])]

def get_model(model_key):
    model_name = MODELS.get(model_key)
    
    if "flan-t5" in model_key:
        return GenerativeNLIModel(model_name)
    else:
        # Keep your existing NLIModel instantiation
        model = NLIModel(model_name)
        # Add BART label mapping mapping if needed. 
        # facebook/bart-large-mnli uses: 0: contradiction, 1: neutral, 2: entailment
        if "bart" in model_key:
            model._model_to_snli_map = {0: 2, 1: 1, 2: 0} 
        return model


if __name__ == "__main__":
    # Self-test with a newer generation DeBERTa model
    test_model_name = "cross-encoder/nli-deberta-v3-base"
    tester = NLIModel(model_name=test_model_name)
    
    p = "A dog is running in the park."
    h = "An animal is moving outside."
    
    pred = tester.predict(p, h)
    print(f"\nTest prediction: {pred} (Expected: 0 for entailment)")