#!/bin/bash
# run_all.sh - Master execution script for the NLI Diagnostic Pipeline

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================"
echo " Starting NLI Partial Input Diagnostic Pipeline"
echo "======================================================"

echo ""
echo ">>> STEP 1: Running Inference & Generating Logs..."
# Modify these lists if you want to run a smaller subset
python run_diagnostics.py \
  --models roberta deberta distilroberta distilbert piri_distilroberta \
  --datasets snli mnli hans anli sick xnli \
  --sample_size 1000

echo ""
echo ">>> STEP 2: Running Console Analysis..."
# This will print your Grand Master Summary and PIRI tables to the console
python analyze.py

echo ""
echo ">>> STEP 3: Generating Visualizations..."
# This regenerates the bar charts and the PIRI vs. HANS scatter plot
python plots.py

echo ""
echo "======================================================"
echo " Pipeline Complete!"
echo " Results saved to results/"
echo " Plots saved to root directory."
echo ""
echo " Command to launch dashboard: streamlit run dashboard.py"
echo "======================================================"