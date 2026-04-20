import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from config import MODELS
from tqdm import tqdm

class NLIModel:
    def __init__(self, model_name="cross-encoder/nli-roberta-base"):
        print(f"Loading model and tokenizer: '{model_name}'...")
        self.model_name = model_name  # <-- NEW: Save the path for the eager clone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fast SDPA model for the evaluation gauntlet
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        if "cross-encoder" in model_name or "piri_model" in model_name:
            self._model_to_snli_map = {0: 2, 1: 0, 2: 1}
        elif "typeform" in model_name:
            self._model_to_snli_map = {0: 0, 1: 1, 2: 2}
        else:
            self._model_to_snli_map = {0: 0, 1: 1, 2: 2}

    def predict(self, premise, hypothesis):
        mapped_preds, mapped_probs = self.predict_batch([premise], [hypothesis], batch_size=1)
        return mapped_preds[0], mapped_probs[0]

    def predict_batch(self, premises, hypotheses, batch_size=4):
        from tqdm import tqdm
        all_mapped_preds, all_mapped_probs = [], []
        for i in tqdm(range(0, len(premises), batch_size), desc="BERT Inference", leave=False):
            p_batch = premises[i:i+batch_size]
            h_batch = hypotheses[i:i+batch_size]
            encodings = self.tokenizer(p_batch, h_batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**encodings).logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                for idx in range(len(preds)):
                    pred, prob = preds[idx], probs[idx]
                    mapped_pred = self._model_to_snli_map.get(pred, pred)
                    mapped_probs = [0.0, 0.0, 0.0]
                    for orig_idx, mapped_idx in self._model_to_snli_map.items():
                        mapped_probs[mapped_idx] = float(prob[orig_idx])
                    all_mapped_preds.append(mapped_pred)
                    all_mapped_probs.append(mapped_probs)
        return all_mapped_preds, all_mapped_probs

    def predict_with_attention(self, premise, hypothesis):
        import gc
        eager_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            attn_implementation="eager",
            output_attentions=True
        ).to(self.device)
        eager_model.eval()
        
        encodings = self.tokenizer(premise, hypothesis, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = eager_model(**encodings)
            pred = torch.argmax(outputs.logits, dim=1).item()
            mapped_pred = self._model_to_snli_map.get(pred, pred)
            
            # Check for BART (encoder_attentions) vs Standard BERT (attentions)
            if hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions is not None:
                last_layer_attn = outputs.encoder_attentions[-1]
            elif hasattr(outputs, 'attentions') and outputs.attentions is not None:
                last_layer_attn = outputs.attentions[-1]
            else:
                del eager_model
                gc.collect()
                raise ValueError(f"Attention extraction blocked by {self.model_name} architecture.")
                
            cls_attention = last_layer_attn.mean(dim=1).squeeze(0)[0].cpu().tolist()
            
        tokens = self.tokenizer.convert_ids_to_tokens(encodings.input_ids[0])
        
        del eager_model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()
            
        return mapped_pred, tokens, cls_attention

class GenerativeNLIModel:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.target_ids = [
            self.tokenizer.encode("yes", add_special_tokens=False)[0],
            self.tokenizer.encode("maybe", add_special_tokens=False)[0],
            self.tokenizer.encode("no", add_special_tokens=False)[0]
        ]

    def predict(self, premise, hypothesis):
        preds, probs = self.predict_batch([premise], [hypothesis], batch_size=1)
        return preds[0], probs[0]

    def predict_batch(self, premises, hypotheses, batch_size=4): # <-- LOWERED BATCH SIZE
        all_preds, all_probs = [], []
        
        # <-- WRAPPED IN TQDM PROGRESS BAR
        for i in tqdm(range(0, len(premises), batch_size), desc="LLM Inference"):
            p_batch = premises[i:i+batch_size]
            h_batch = hypotheses[i:i+batch_size]
            prompts = [f"Premise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis? Answer 'yes', 'maybe', or 'no'." for p, h in zip(p_batch, h_batch)]
            
            inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
                first_token_logits = outputs.scores[0]
                
                for b_idx in range(len(p_batch)):
                    target_logits = torch.tensor([first_token_logits[b_idx][tid] for tid in self.target_ids])
                    probs = F.softmax(target_logits, dim=0).cpu().numpy()
                    pred_idx = torch.argmax(target_logits).item()
                    all_preds.append(pred_idx)
                    all_probs.append([float(probs[0]), float(probs[1]), float(probs[2])])
                    
        return all_preds, all_probs

def get_model(model_key):
    model_name = MODELS.get(model_key)
    if "flan-t5" in model_key:
        return GenerativeNLIModel(model_name)
    else:
        model = NLIModel(model_name)
        if "bart" in model_key:
            model._model_to_snli_map = {0: 2, 1: 1, 2: 0} 
        return model