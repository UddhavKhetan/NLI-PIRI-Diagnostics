import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import MODELS, DATASETS

class PIRIMitigationTrainer:
    def __init__(self, model_key: str, alpha: float, lr: float, device: str):
        self.model_config = MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.hf_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_config.hf_id)
        self.model.to(device)
        self.alpha = alpha
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train_step(self, premises: list, hypotheses: list, labels: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad()
        labels = labels.to(self.device)

        # Forward Pass 1: Full Input
        inputs_full = self.tokenizer(premises, hypotheses, return_tensors='pt', padding=True, truncation=True).to(self.device)
        logits_full = self.model(**inputs_full).logits

        # Forward Pass 2: Hypothesis-Only (Premise omitted/empty)
        empty_premises = [""] * len(hypotheses)
        inputs_hyp = self.tokenizer(empty_premises, hypotheses, return_tensors='pt', padding=True, truncation=True).to(self.device)
        logits_hyp = self.model(**inputs_hyp).logits

        # Cross Entropy Loss
        loss_ce = F.cross_entropy(logits_full, labels)

        # KL Divergence Loss: Regularizing the full model distribution against the biased hypothesis-only distribution.
        # F.kl_div expects log_softmax for input and softmax for target.
        # This penalizes the model when the full input relies too heavily on the hypothesis.
        loss_kl = F.kl_div(
            F.log_softmax(logits_full, dim=-1),
            F.softmax(logits_hyp.detach(), dim=-1),
            reduction='batchmean'
        )

        total_loss = loss_ce + (self.alpha * loss_kl)
        total_loss.backward()
        self.optimizer.step()

        return loss_ce.item(), loss_kl.item(), total_loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PIRI Mitigation Training")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of the KL-divergence penalty.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds to run.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    print(f"Initialized mitigation experiment for {args.model} on {args.dataset} with alpha={args.alpha}")
    # Integration with data loaders goes here