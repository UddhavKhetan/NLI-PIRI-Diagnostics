from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    hf_id: str
    arch_type: str  # 'encoder', 'encoder-decoder', 'decoder'
    special_tokens_required: bool = False

@dataclass
class DatasetConfig:
    bias_type: str
    num_classes: int = 3

MODELS: Dict[str, ModelConfig] = {
    "roberta": ModelConfig("roberta-base", "encoder"),
    "deberta": ModelConfig("microsoft/deberta-base", "encoder"),
    "distilroberta": ModelConfig("distilroberta-base", "encoder"),
    "distilbert": ModelConfig("distilbert-base-uncased", "encoder"),
    "bart": ModelConfig("facebook/bart-large-mnli", "encoder-decoder"),
    "flan-t5": ModelConfig("google/flan-t5-base", "encoder-decoder", special_tokens_required=True)
}

DATASETS: Dict[str, DatasetConfig] = {
    "mnli": DatasetConfig("baseline_hypothesis_bias", 3),
    "snli": DatasetConfig("lexical_overlap", 3),
    "hans": DatasetConfig("adversarial_heuristics", 2),
    "anli": DatasetConfig("adversarial_robustness", 3),
    "xnli": DatasetConfig("multilingual_probing", 3),
    "sick": DatasetConfig("compositional_logic", 3)
}

ABLATION_STRATEGIES: List[str] = ["empty", "mask", "random", "neutral"]