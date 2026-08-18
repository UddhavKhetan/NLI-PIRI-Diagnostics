import random

def ablate_empty(text: str, **kwargs) -> str:
    return ""

def ablate_mask(text: str, mask_token: str = "[MASK]", **kwargs) -> str:
    words = text.split()
    return " ".join([mask_token] * len(words))

def ablate_random(text: str, **kwargs) -> str:
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

def ablate_neutral(text: str, **kwargs) -> str:
    return "This is a neutral and uninformative statement."

ABLATION_MAP = {
    "empty": ablate_empty,
    "mask": ablate_mask,
    "random": ablate_random,
    "neutral": ablate_neutral
}

def apply_ablation(premise: str, hypothesis: str, condition: str, strategy: str, mask_token: str = "[MASK]") -> tuple:
    if strategy not in ABLATION_MAP:
        raise ValueError(f"Unknown ablation strategy: {strategy}")
        
    ablation_func = ABLATION_MAP[strategy]
    
    if condition == "full":
        return premise, hypothesis
    elif condition == "premise-only":
        return premise, ablation_func(hypothesis, mask_token=mask_token)
    elif condition == "hypothesis-only":
        return ablation_func(premise, mask_token=mask_token), hypothesis
    else:
        raise ValueError(f"Unknown condition: {condition}")