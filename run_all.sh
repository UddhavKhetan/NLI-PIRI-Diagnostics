#!/bin/bash

# Configuration
MODELS="roberta deberta distilroberta distilbert bart flan-t5"
DATASETS="snli mnli hans anli sick xnli"
ABLATIONS=("empty" "mask" "neutral" "random")
SEEDS="42 43 44"
SAMPLE_SIZE=100

echo "Starting full NLI diagnostic gauntlet..."
echo "Models: $MODELS"
echo "Datasets: $DATASETS"
echo "Seeds: $SEEDS"
echo "Sample Size: $SAMPLE_SIZE"

# Loop through each ablation strategy
for ablation in "${ABLATIONS[@]}"; do
    echo ""
    echo "====================================================="
    echo ">> Running ablation strategy: $ablation"
    echo "====================================================="
    
    python run_diagnostics.py \
        --models $MODELS \
        --datasets $DATASETS \
        --sample_size $SAMPLE_SIZE \
        --seeds $SEEDS \
        --ablation_strategy $ablation
done

echo ""
echo "====================================================="
echo "All combinations populated. You can now launch the dashboard."
echo "Command: streamlit run dashboard.py"
