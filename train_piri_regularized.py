import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.optim import AdamW
import time
import os

def load_data_splits(train_size=10000, val_size=1000):
    print("Loading SNLI dataset subsets...")
    dataset = load_dataset("snli")
    
    # Filter out consensus failures (-1)
    train_ds = dataset['train'].filter(lambda x: x['label'] != -1).shuffle(seed=42).select(range(train_size))
    val_ds = dataset['validation'].filter(lambda x: x['label'] != -1).shuffle(seed=42).select(range(val_size))
    
    return train_ds, val_ds

def train():
    # --- Hyperparameters ---
    model_name = "cross-encoder/nli-distilroberta-base"
    batch_size = 16
    epochs = 3
    lr = 2e-5
    lambda_kl = 0.1  # Our PIRI penalty weight
    
    # --- Setup Device ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model & Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # --- Data Loaders ---
    train_ds, val_ds = load_data_splits()
    
    # Custom collate to just pass raw text strings to the training loop
    def collate_fn(batch):
        premises = [item['premise'] for item in batch]
        hypotheses = [item['hypothesis'] for item in batch]
        
        # MAP LABELS: SNLI (0=Ent, 1=Neu, 2=Con) -> CrossEncoder (0=Con, 1=Ent, 2=Neu)
        label_map = {0: 1, 1: 2, 2: 0}
        labels = torch.tensor([label_map[item['label']] for item in batch], dtype=torch.long)
        return premises, hypotheses, labels

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

    # --- Training Loop ---
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_ce, total_kl = 0, 0, 0
        start_time = time.time()
        
        for step, (premises, hypotheses, labels) in enumerate(train_loader):
            labels = labels.to(device)
            optimizer.zero_grad()

            # 1. Full Input Forward Pass
            full_encodings = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt").to(device)
            logits_full = model(**full_encodings).logits
            
            # 2. Hypothesis-Only Forward Pass (Empty Premise)
            empty_premises = [""] * len(premises)
            hyp_encodings = tokenizer(empty_premises, hypotheses, padding=True, truncation=True, return_tensors="pt").to(device)
            logits_hyp = model(**hyp_encodings).logits

            # 3. Calculate Losses
            # Standard classification loss on the full input
            ce_loss = F.cross_entropy(logits_full, labels)
            
            # KL Divergence: maximize difference between full and hyp-only predictions
            # PyTorch kl_div: input must be log-probabilities, target must be probabilities
            p_full = F.softmax(logits_full, dim=-1)
            log_p_hyp = F.log_softmax(logits_hyp, dim=-1)
            kl_loss = F.kl_div(log_p_hyp, p_full, reduction='batchmean')
            
            # We SUBTRACT the KL loss to maximize the divergence
            loss = ce_loss - (lambda_kl * kl_loss)

            # 4. Backward Pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_kl += kl_loss.item()

            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Total Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, KL: {kl_loss.item():.4f})")

        # --- Validation Loop (Standard Accuracy on Full Inputs) ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for premises, hypotheses, labels in val_loader:
                labels = labels.to(device)
                encodings = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt").to(device)
                logits = model(**encodings).logits
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_acc = correct / total * 100
        epoch_time = time.time() - start_time
        print(f"\n>>> Epoch {epoch+1} Summary | Val Acc: {val_acc:.1f}% | Time: {epoch_time:.0f}s")
        print("-" * 60)

    # --- Save the Model ---
    output_dir = "./piri_model_distilroberta"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving fine-tuned model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    train()