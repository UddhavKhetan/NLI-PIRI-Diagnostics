from datasets import load_dataset

def load_and_sample_snli(sample_size=20):
    print("Loading SNLI dataset (validation split)...")
    # Load only the validation split to save time and bandwidth
    dataset = load_dataset("snli", split="validation")
    
    print(f"Original size: {len(dataset)}")
    
    # SNLI contains some labels marked as -1 (annotators couldn't agree). 
    # We must filter these out to avoid crashing the model later.
    dataset = dataset.filter(lambda x: x["label"] != -1)
    
    # Take a small sample for rapid testing
    sampled_dataset = dataset.select(range(sample_size))
    print(f"Sampled size: {len(sampled_dataset)}\n")
    
    # Print a few examples to verify the structure
    print("--- First 3 Examples ---")
    for i in range(3):
        example = sampled_dataset[i]
        print(f"Example {i+1}:")
        print(f"  Premise:    {example['premise']}")
        print(f"  Hypothesis: {example['hypothesis']}")
        print(f"  Label:      {example['label']} (0=entailment, 1=neutral, 2=contradiction)\n")

    return sampled_dataset

if __name__ == "__main__":
    # This block ensures the function runs when we execute the script
    load_and_sample_snli()