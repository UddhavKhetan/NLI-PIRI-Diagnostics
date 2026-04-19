#!/bin/bash

# Configuration
MODELS=("roberta" "deberta" "distilroberta" "distilbert" "bart" "flan-t5")
DATASETS=("snli" "mnli" "hans" "anli" "sick" "xnli")
ABLATIONS=("empty" "mask" "neutral" "random")
SEEDS="42 43 44"
SAMPLE_SIZE=500

# Setup directories for isolation
mkdir -p results logs

echo "Starting Resilient NLI Gauntlet on Apple Silicon..."

# Iterate granularly to allow strict resume capabilities
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            
            CSV_PATH="results/${dataset}_${model}_${ablation}_results.csv"
            LOG_PATH="logs/${dataset}_${model}_${ablation}.log"

            # 1. Skip Completed Jobs (Resume capability)
            if [ -f "$CSV_PATH" ]; then
                # Quick check to ensure it's not an empty/corrupt file
                if [ -s "$CSV_PATH" ]; then
                    echo "[SKIP] Data exists for: $model | $dataset | $ablation"
                    continue
                fi
            fi

            echo "[RUNNING] $model | $dataset | $ablation -> Logging to $LOG_PATH"
            
            # 2. Execute & Log. Catch failures without killing the loop.
            python run_diagnostics.py \
                --models "$model" \
                --datasets "$dataset" \
                --sample_size $SAMPLE_SIZE \
                --seeds $SEEDS \
                --ablation_strategy "$ablation" > "$LOG_PATH" 2>&1
            
            # Check exit status
            if [ $? -ne 0 ]; then
                echo "[ERROR] Failed: $model | $dataset | $ablation. See $LOG_PATH"
                # If it crashed, remove the potentially corrupted partial CSV
                rm -f "$CSV_PATH"
            else
                echo "[SUCCESS] Completed: $model | $dataset | $ablation"
            fi
            
            # 3. Avoid Overloading M4 Unified Memory
            # Give PyTorch/MPS a moment to garbage collect before loading the next model
            sleep 2
            
        done
    done
done

echo "====================================================="
echo "Gauntlet Complete. Launch dashboard with: streamlit run dashboard.py"
