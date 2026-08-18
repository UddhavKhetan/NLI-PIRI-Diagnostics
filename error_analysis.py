# src/error_analysis.py
import csv
import os

def build_confusion_matrix(y_true, y_pred, num_classes):
    """
    Builds a confusion matrix manually.
    Rows represent True labels, Columns represent Predicted labels.
    """
    matrix = []
    for _ in range(num_classes):
        row = []
        for _ in range(num_classes):
            row.append(0)
        matrix.append(row)
        
    for i in range(len(y_true)):
        actual = y_true[i]
        predicted = y_pred[i]
        matrix[actual][predicted] += 1
        
    return matrix

def compute_per_label_accuracy(confusion_matrix, num_classes):
    """
    Computes accuracy for each individual label based on the confusion matrix.
    """
    per_label_acc = []
    for cls in range(num_classes):
        total_actual = 0
        for col in range(num_classes):
            total_actual += confusion_matrix[cls][col]
            
        if total_actual == 0:
            per_label_acc.append(0.0)
        else:
            correct = confusion_matrix[cls][cls]
            per_label_acc.append(correct / total_actual)
            
    return per_label_acc

def save_confusion_matrix_to_csv(matrix, filename, label_names):
    """
    Saves the nested list confusion matrix to a CSV file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        header = ["True_Label"] + label_names
        writer.writerow(header)
        
        # Write rows
        for i in range(len(matrix)):
            row = [label_names[i]] + matrix[i]
            writer.writerow(row)