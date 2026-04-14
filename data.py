import pandas as pd
from datasets import load_dataset, Dataset
from config import DATASETS

def load_snli_data(sample_size=1000, seed=42):
    """
    Loads and filters the SNLI validation split.
    """
    print(f"Loading SNLI validation split (sample size: {sample_size}, seed: {seed})...")
    dataset = load_dataset("snli", split="validation")
    
    # Filter out lack of consensus labels (-1)
    dataset = dataset.filter(lambda x: x["label"] != -1)
    
    # Apply sampling if specified
    if sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
        
    return dataset

def load_mnli_matched(sample_size=1000):
    """
    Loads and filters the MNLI validation_matched split.
    """
    print(f"Loading MNLI validation_matched split (sample size: {sample_size})...")
    # We use the GLUE benchmark version of MNLI
    dataset = load_dataset("glue", "mnli", split="validation_matched")
    
    # Filter out lack of consensus labels (-1)
    dataset = dataset.filter(lambda x: x["label"] != -1)
    
    if sample_size:
        dataset = dataset.select(range(sample_size))
        
    return dataset

def load_hans(sample_size=1000):
    """
    Loads HANS manually from the official GitHub repo to bypass Hugging Face script errors.
    """
    print(f"Downloading/Loading HANS dataset manually (sample size: {sample_size})...")
    url = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
    
    # Read the raw tab-separated file
    df = pd.read_csv(url, sep='\t')
    
    # Rename columns to match SNLI/MNLI expectations
    df = df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis"})
    
    # Map string labels to our integer schema
    # 'entailment' -> 0, 'non-entailment' -> 2
    df['label'] = df['gold_label'].apply(lambda x: 0 if x == 'entailment' else 2)
    
    # Convert back to a Hugging Face Dataset object
    dataset = Dataset.from_pandas(df)
    
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        
    return dataset

def load_xnli(language="en", sample_size=1000):
    """
    Loads the XNLI validation dataset for a specific language.
    Languages include: 'en', 'fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'ar', 'vi', 'th', 'zh', 'hi', 'sw', 'ur'
    """
    print(f"Loading XNLI ({language}) validation split (sample size: {sample_size})...")
    dataset = load_dataset("xnli", language, split="validation")
    
    # XNLI labels exactly match SNLI: 0=entailment, 1=neutral, 2=contradiction
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        
    return dataset

def load_anli(round_num=3, sample_size=1000):
    """
    Loads the ANLI dataset for a specific round (1, 2, or 3).
    We use the dev split, which are named 'dev_r1', 'dev_r2', 'dev_r3'.
    """
    print(f"Loading ANLI Round {round_num} dev split (sample size: {sample_size})...")
    split_name = f"dev_r{round_num}"
    dataset = load_dataset("anli", split=split_name)
    
    # ANLI dev_r3 has exactly 1200 examples, so we cap it safely
    if sample_size:
        actual_size = min(sample_size, len(dataset))
        dataset = dataset.shuffle(seed=42).select(range(actual_size))
        
    return dataset

def load_sick(sample_size=1000):
    """
    Loads SICK manually from a raw TSV to bypass Hugging Face script errors.
    Requires strict compositional reasoning without crowd-worker artifacts.
    """
    print(f"Downloading/Loading SICK dataset manually (sample size: {sample_size})...")
    url = "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt"
    
    # Read the raw tab-separated file
    df = pd.read_csv(url, sep='\t')
    
    # Rename columns to match SNLI/MNLI expectations
    df = df.rename(columns={"sentence_A": "premise", "sentence_B": "hypothesis"})
    
    # Map string labels to our integer schema
    label_map = {'ENTAILMENT': 0, 'NEUTRAL': 1, 'CONTRADICTION': 2}
    df['label'] = df['entailment_judgment'].map(label_map)
    
    # Clean any malformed rows and cast to int
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    # Convert back to a Hugging Face Dataset object
    dataset = Dataset.from_pandas(df)
    
    if sample_size:
        actual_size = min(sample_size, len(dataset))
        dataset = dataset.shuffle(seed=42).select(range(actual_size))
        
    return dataset


def get_dataset(dataset_key, sample_size=None, seed=42, **kwargs):
    """Router function to fetch the correct dataset based on string key."""
    dataset_key = dataset_key.lower()
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS.keys())}")
        
    # Use provided sample size, fallback to config default
    size = sample_size if sample_size is not None else DATASETS[dataset_key]["default_sample"]
    
    # Merge any default kwargs from config with explicitly passed kwargs
    cfg_kwargs = DATASETS[dataset_key].get("default_kwargs", {})
    merged_kwargs = {**cfg_kwargs, **kwargs}

    if dataset_key == "snli": return load_snli_data(sample_size=size, seed=seed)
    elif dataset_key == "mnli": return load_mnli_matched(sample_size=size)
    elif dataset_key == "hans": return load_hans(sample_size=size)
    elif dataset_key == "xnli": return load_xnli(sample_size=size, **merged_kwargs)
    elif dataset_key == "anli": return load_anli(sample_size=size, **merged_kwargs)
    elif dataset_key == "sick": return load_sick(sample_size=size)

if __name__ == "__main__":
    # Quick self-test to ensure the module works independently
    ds = load_snli_data(5)
    print(f"\nTest successful. Loaded {len(ds)} examples.")
    print(f"Example 0 Label: {ds[0]['label']}")