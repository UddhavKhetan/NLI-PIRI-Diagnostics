# config.py

# Map human-readable CLI keys to exact model paths
MODELS = {
    "roberta": "cross-encoder/nli-roberta-base",
    "deberta": "cross-encoder/nli-deberta-v3-base",
    "distilroberta": "cross-encoder/nli-distilroberta-base",
    "distilbert": "typeform/distilbert-base-uncased-mnli",
    "mdeberta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "bart": "facebook/bart-large-mnli",
    "flan-t5": "google/flan-t5-small",
    "piri_distilroberta": "./piri_model_a10" # Custom model
}

# Define datasets and their default behaviors
DATASETS = {
    "snli": {"default_sample": 1000},
    "mnli": {"default_sample": 1000},
    "hans": {"default_sample": 1000},
    "xnli": {"default_sample": 1000, "default_kwargs": {"language": "en"}},
    "anli": {"default_sample": 1000, "default_kwargs": {"round_num": 3}},
    "sick": {"default_sample": 1000}
}