import torch
from captum.attr import LayerIntegratedGradients

class NLIExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _get_embedding_layer(self):
        """Dynamically fetch the embedding layer based on the HF architecture."""
        if hasattr(self.model, 'roberta'):
            return self.model.roberta.embeddings.word_embeddings
        elif hasattr(self.model, 'deberta'):
            return self.model.deberta.embeddings.word_embeddings
        elif hasattr(self.model, 'distilbert'):
            return self.model.distilbert.embeddings.word_embeddings
        else:
            # Fallback to the first child module's embedding
            first_module = list(self.model.named_children())[0][1]
            if hasattr(first_module, 'embeddings'):
                return first_module.embeddings.word_embeddings
            raise ValueError("Unsupported architecture for LayerIntegratedGradients extraction.")

    def explain_integrated_gradients(self, premise: str, hypothesis: str, target_class: int):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        # Baseline: sequence of padding/zero tokens
        ref_input_ids = torch.zeros_like(input_ids).to(self.device)
        
        layer = self._get_embedding_layer()
        lig = LayerIntegratedGradients(self.model, layer)
        
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=ref_input_ids,
            target=target_class,
            return_convergence_delta=True
        )
        
        # Summarize across embedding dimensions
        scores = attributions.sum(dim=-1).squeeze(0)
        scores = scores / torch.norm(scores) # Normalize
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return tokens, scores.detach().cpu().numpy()