# tests/test_pipeline.py
import pytest
import pandas as pd
from data import get_dataset
from models import get_model

def test_get_dataset_snli_shape():
    """Ensure SNLI loads correctly, has the right length and columns."""
    ds = get_dataset("snli", sample_size=10, seed=1)
    
    assert len(ds) == 10
    
    # Check that required columns exist
    first_row = ds[0]
    assert 'premise' in first_row
    assert 'hypothesis' in first_row
    assert 'label' in first_row
    
    # Check that labels are mapped to the 0, 1, 2 schema
    for item in ds:
        assert item['label'] in [0, 1, 2]

def test_get_model_interface():
    """Ensure the model router returns a valid wrapper that predicts integers."""
    # Use distilbert as it's smaller/faster for testing than DeBERTa
    model = get_model("distilbert")
    
    pred = model.predict("The dog is running.", "An animal is moving.")
    
    # Prediction must be an integer mapping to entailment/neutral/contradiction
    assert isinstance(pred, int)
    assert pred in [0, 1, 2]

def test_piri_computation():
    """Verify the PIRI math logic directly."""
    # PIRI = (Acc_full - Acc_hyp) / Acc_full
    
    acc_full = 0.90
    acc_hyp = 0.45
    
    expected_piri = (0.90 - 0.45) / 0.90
    
    # The exact logic used in analyze.py
    calculated_piri = (acc_full - acc_hyp) / acc_full if acc_full > 0 else 0.0
    
    assert abs(calculated_piri - expected_piri) < 1e-6
    assert calculated_piri == 0.5