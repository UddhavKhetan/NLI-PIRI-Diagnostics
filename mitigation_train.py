# mitigation_train.py
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_presets(model_choice):
    """
    Defines explicit, hardcoded presets for larger models requested by reviewers.
    """
    presets = {
        "roberta": {
            "hf_id": "roberta-large",
            "dataset": "mnli",
            "alpha": 0.5,
            "lr": 1e-5,
            "batch_size": 16,
            "epochs": 3,
            "seeds": [42, 43, 44]
        },
        "deberta": {
            "hf_id": "microsoft/deberta-v3-large",
            "dataset": "mnli",
            "alpha": 0.7,
            "lr": 8e-6,
            "batch_size": 8,
            "epochs": 3,
            "seeds": [42, 43, 44]
        },
        "bart_large_mnli": {
            "hf_id": "facebook/bart-large-mnli",
            "dataset": "mnli",
            "alpha": 0.5,
            "lr": 1e-5,
            "batch_size": 16,
            "epochs": 2,
            "seeds": [42, 43, 44]
        },
        "flan_t5_small": {
            "hf_id": "google/flan-t5-small",
            "dataset": "snli",
            "alpha": 0.3,
            "lr": 5e-5,
            "batch_size": 32,
            "epochs": 5,
            "seeds": [42, 43, 44]
        }
    }
    return presets.get(model_choice, presets["roberta"])

def dummy_dataloader(batch_size):
    return [{"premise": ["Premise 1"], "hypothesis": ["Hypothesis 1"], "labels": torch.tensor([0])}]

def train_mitigation(preset_name):
    config = get_presets(preset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting mitigation for {preset_name} with config: {config}")

    for seed in config["seeds"]:
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(config["hf_id"])
        model = AutoModelForSequenceClassification.from_pretrained(config["hf_id"])
        model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        dataloader = dummy_dataloader(config["batch_size"])
        
        for epoch in range(config["epochs"]):
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Explicit unoptimized dual forward pass
                inputs_full = tokenizer(batch["premise"], batch["hypothesis"], return_tensors='pt', padding=True).to(device)
                empty_premises = [""] * len(batch["hypothesis"])
                inputs_hyp = tokenizer(empty_premises, batch["hypothesis"], return_tensors='pt', padding=True).to(device)
                
                labels = batch["labels"].to(device)
                
                logits_full = model(**inputs_full).logits
                logits_hyp = model(**inputs_hyp).logits
                
                loss_ce = F.cross_entropy(logits_full, labels)
                loss_kl = F.kl_div(
                    F.log_softmax(logits_full, dim=-1),
                    F.softmax(logits_hyp.detach(), dim=-1),
                    reduction='batchmean'
                )
                
                total_loss = loss_ce + (config["alpha"] * loss_kl)
                total_loss.backward()
                optimizer.step()
                
            print(f"Seed {seed}, Epoch {epoch+1} Complete. Loss: {total_loss.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["roberta", "deberta", "bart_large_mnli", "flan_t5_small"])
    args = parser.parse_args()
    
    train_mitigation(args.model)