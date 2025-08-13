import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import os
import sys
import argparse
from pathlib import Path

map_emotion = {}

# This is for SER (because I trained using small names for emotions)
# map_emotion['neutral'] = 'neu'
# map_emotion['happy'] = 'hap'
# map_emotion['sad'] = 'sad'
# map_emotion['angry'] = 'ang'

# This is for SLM
map_emotion['neutral'] = 'neutral'
map_emotion['happy'] = 'happy'
map_emotion['sad'] = 'sad'
map_emotion['angry'] = 'angry'

def extract_emotions(df):
    """Extract true and proxy emotions from filenames"""
    true_emotions = []
    proxy_emotions = []

    try:
        predicted_emotions = df['predicted_emotion'].str.lower().tolist()
        print("getting from predicted_emotion")
    except:
        try:
            predicted_emotions = df['emotion'].str.lower().tolist()
            print("getting from emotion")
        except:
            try:
                predicted_emotions = df['output'].str.lower().tolist()
                print("getting from output")
            except:
                print("Error: 'predicted_emotion' column not found in CSV.")
                sys.exit(1)
    
    fpath_col = "filename"
    if(fpath_col not in df.columns):
        fpath_col = "file_path"
        print("Using 'file_path' column instead of 'filename'")

    for filename in df[fpath_col]:
        parts = filename.split('_')
        
        # True emotion is at position -3 (before the last underscore and file extension)
        true_emotion = parts[-3] if len(parts) >= 3 else 'unknown'
        true_emotions.append(map_emotion.get(true_emotion, true_emotion))
        
        # Proxy emotion is at position 1
        proxy_emotion = parts[1] if len(parts) >= 2 else 'unknown'
        proxy_emotions.append(map_emotion.get(proxy_emotion, proxy_emotion))
    
    return true_emotions, proxy_emotions, predicted_emotions

def create_confusion_matrix_plot(y_true, y_pred, labels, save_path, title, xlabel, ylabel):
    """Create and save a beautiful confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 12})
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

def calculate_metrics(y_true, y_pred, labels):
    """Calculate accuracy and F1 macro score"""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    # Get per-class metrics
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    
    return accuracy, f1_macro, report

def save_metrics(accuracy_true, f1_macro_true, report_true, cm_true, 
                 accuracy_proxy, f1_macro_proxy, report_proxy, cm_proxy, 
                 labels, save_path):
    """Save metrics to a text file"""
    with open(save_path, 'w') as f:
        f.write("=== Emotion Prediction Analysis Metrics ===\n\n")
        
        # True vs Predicted metrics
        f.write("*** TRUE EMOTION vs PREDICTED EMOTION ***\n")
        f.write(f"Overall Accuracy: {accuracy_true:.4f} ({accuracy_true*100:.2f}%)\n")
        f.write(f"F1 Macro Score: {f1_macro_true:.4f}\n\n")
        
        f.write("=== Confusion Matrix ===\n")
        f.write("Rows: True Emotion (from filename)\n")
        f.write("Columns: Predicted Emotion\n\n")
        
        # Write confusion matrix with labels
        header = "True\\Predicted\t" + "\t".join(labels)
        f.write(header + "\n")
        
        for i, label in enumerate(labels):
            row = f"{label}\t\t" + "\t".join(str(cm_true[i, j]) for j in range(len(labels)))
            f.write(row + "\n")
        
        f.write("\n=== Detailed Classification Report ===\n")
        f.write(report_true)
        
        # Proxy vs Predicted metrics
        f.write("\n\n*** PROXY EMOTION vs PREDICTED EMOTION ***\n")
        f.write(f"Overall Accuracy: {accuracy_proxy:.4f} ({accuracy_proxy*100:.2f}%)\n")
        f.write(f"F1 Macro Score: {f1_macro_proxy:.4f}\n\n")
        
        f.write("=== Confusion Matrix ===\n")
        f.write("Rows: Proxy Emotion (from filename)\n")
        f.write("Columns: Predicted Emotion\n\n")
        
        # Write confusion matrix with labels
        header = "Proxy\\Predicted\t" + "\t".join(labels)
        f.write(header + "\n")
        
        for i, label in enumerate(labels):
            row = f"{label}\t\t" + "\t".join(str(cm_proxy[i, j]) for j in range(len(labels)))
            f.write(row + "\n")
        
        f.write("\n=== Detailed Classification Report ===\n")
        f.write(report_proxy)

def main(csv_filepath, output_path):
    """Main function to process emotion predictions"""
    # Read CSV file
    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print(df.columns)

    # Extract emotions
    true_emotions, proxy_emotions, predicted_emotions = extract_emotions(df)
    
    # Get unique labels
    all_emotions = list(set(true_emotions + proxy_emotions + predicted_emotions))
    all_emotions.sort()  # Sort for consistent ordering
    
    # Create output directory
    csv_filename = Path(csv_filepath).stem
    output_dir = Path(output_path) / csv_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create confusion matrix for True vs Predicted
    cm_true_path = output_dir / "confusion_matrix_true_vs_predicted.png"
    cm_true = create_confusion_matrix_plot(
        true_emotions, predicted_emotions, all_emotions, cm_true_path,
        'Confusion Matrix: True Emotion vs Predicted Emotion',
        'Predicted Emotion',
        'True Emotion (from filename)'
    )
    print(f"True vs Predicted confusion matrix saved to: {cm_true_path}")
    
    # Create confusion matrix for Proxy vs Predicted
    cm_proxy_path = output_dir / "confusion_matrix_proxy_vs_predicted.png"
    cm_proxy = create_confusion_matrix_plot(
        proxy_emotions, predicted_emotions, all_emotions, cm_proxy_path,
        'Confusion Matrix: Proxy Emotion vs Predicted Emotion',
        'Predicted Emotion',
        'Proxy Emotion (from filename)'
    )
    print(f"Proxy vs Predicted confusion matrix saved to: {cm_proxy_path}")
    
    # Calculate metrics for True vs Predicted
    accuracy_true, f1_macro_true, report_true = calculate_metrics(
        true_emotions, predicted_emotions, all_emotions
    )
    
    # Calculate metrics for Proxy vs Predicted
    accuracy_proxy, f1_macro_proxy, report_proxy = calculate_metrics(
        proxy_emotions, predicted_emotions, all_emotions
    )
    
    # Save metrics
    metrics_path = output_dir / "metrics.txt"
    save_metrics(
        accuracy_true, f1_macro_true, report_true, cm_true,
        accuracy_proxy, f1_macro_proxy, report_proxy, cm_proxy,
        all_emotions, metrics_path
    )
    print(f"Metrics saved to: {metrics_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"True vs Predicted - Accuracy: {accuracy_true:.4f} ({accuracy_true*100:.2f}%), F1 Macro: {f1_macro_true:.4f}")
    print(f"Proxy vs Predicted - Accuracy: {accuracy_proxy:.4f} ({accuracy_proxy*100:.2f}%), F1 Macro: {f1_macro_proxy:.4f}")
    print(f"\nAnalysis complete! Results saved in: {output_dir}")



    ## Now for explicit emotions
    # Read CSV file
    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
        
    fpath_col = "filename"
    if(fpath_col not in df.columns):
        fpath_col = "file_path"
        print("Using 'file_path' column instead of 'filename'")

    filenames_ = df[fpath_col]
    is_explicit = []
    for f in filenames_:
        if "explicit" in f:
            is_explicit.append(1)
        elif "implicit" in f:
            is_explicit.append(0)
        else:
            is_explicit.append(-1)

    df["is_explicit"] = is_explicit
    # TODO: confirm what to do with -1 ""neutral""
    df_explicit = df[df["is_explicit"] == 1].reset_index(drop=True)
    df_implicit = df[df["is_explicit"] == 0].reset_index(drop=True)
    df_other = df[df["is_explicit"] == -1].reset_index(drop=True)

    print(f"df_explicit shape: {df_explicit.shape}")
    print(f"df_implicit shape: {df_implicit.shape}")
    print(f"df_other shape: {df_other.shape}")

    # Extract emotions
    true_emotions, proxy_emotions, predicted_emotions = extract_emotions(df_explicit)
    
    # Get unique labels
    all_emotions = list(set(true_emotions + proxy_emotions + predicted_emotions))
    all_emotions.sort()  # Sort for consistent ordering
    
    # Create output directory
    csv_filename = Path(csv_filepath).stem
    output_dir = Path(output_path) / csv_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create confusion matrix for True vs Predicted
    cm_true_path = output_dir / "explicit_confusion_matrix_true_vs_predicted.png"
    cm_true = create_confusion_matrix_plot(
        true_emotions, predicted_emotions, all_emotions, cm_true_path,
        'Confusion Matrix: True Emotion vs Predicted Emotion',
        'Predicted Emotion',
        'True Emotion (from filename)'
    )
    print(f"True vs Predicted confusion matrix saved to: {cm_true_path}")
    
    # Create confusion matrix for Proxy vs Predicted
    cm_proxy_path = output_dir / "explicit_confusion_matrix_proxy_vs_predicted.png"
    cm_proxy = create_confusion_matrix_plot(
        proxy_emotions, predicted_emotions, all_emotions, cm_proxy_path,
        'Confusion Matrix: Proxy Emotion vs Predicted Emotion',
        'Predicted Emotion',
        'Proxy Emotion (from filename)'
    )
    print(f"Proxy vs Predicted confusion matrix saved to: {cm_proxy_path}")
    
    # Calculate metrics for True vs Predicted
    accuracy_true, f1_macro_true, report_true = calculate_metrics(
        true_emotions, predicted_emotions, all_emotions
    )
    
    # Calculate metrics for Proxy vs Predicted
    accuracy_proxy, f1_macro_proxy, report_proxy = calculate_metrics(
        proxy_emotions, predicted_emotions, all_emotions
    )
    
    # Save metrics
    metrics_path = output_dir / "metrics_explicit.txt"
    save_metrics(
        accuracy_true, f1_macro_true, report_true, cm_true,
        accuracy_proxy, f1_macro_proxy, report_proxy, cm_proxy,
        all_emotions, metrics_path
    )
    print(f"Metrics saved to: {metrics_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"True vs Predicted - Accuracy: {accuracy_true:.4f} ({accuracy_true*100:.2f}%), F1 Macro: {f1_macro_true:.4f}")
    print(f"Proxy vs Predicted - Accuracy: {accuracy_proxy:.4f} ({accuracy_proxy*100:.2f}%), F1 Macro: {f1_macro_proxy:.4f}")
    print(f"\nAnalysis complete! Results saved in: {output_dir}")


    ## Now for implicit emotions
    # Extract emotions
    true_emotions, proxy_emotions, predicted_emotions = extract_emotions(df_implicit)
    
    # Get unique labels
    all_emotions = list(set(true_emotions + proxy_emotions + predicted_emotions))
    all_emotions.sort()  # Sort for consistent ordering
    
    # Create output directory
    csv_filename = Path(csv_filepath).stem
    output_dir = Path(output_path) / csv_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create confusion matrix for True vs Predicted
    cm_true_path = output_dir / "implicit_confusion_matrix_true_vs_predicted.png"
    cm_true = create_confusion_matrix_plot(
        true_emotions, predicted_emotions, all_emotions, cm_true_path,
        'Confusion Matrix: True Emotion vs Predicted Emotion',
        'Predicted Emotion',
        'True Emotion (from filename)'
    )
    print(f"True vs Predicted confusion matrix saved to: {cm_true_path}")
    
    # Create confusion matrix for Proxy vs Predicted
    cm_proxy_path = output_dir / "implicit_confusion_matrix_proxy_vs_predicted.png"
    cm_proxy = create_confusion_matrix_plot(
        proxy_emotions, predicted_emotions, all_emotions, cm_proxy_path,
        'Confusion Matrix: Proxy Emotion vs Predicted Emotion',
        'Predicted Emotion',
        'Proxy Emotion (from filename)'
    )
    print(f"Proxy vs Predicted confusion matrix saved to: {cm_proxy_path}")
    
    # Calculate metrics for True vs Predicted
    accuracy_true, f1_macro_true, report_true = calculate_metrics(
        true_emotions, predicted_emotions, all_emotions
    )
    
    # Calculate metrics for Proxy vs Predicted
    accuracy_proxy, f1_macro_proxy, report_proxy = calculate_metrics(
        proxy_emotions, predicted_emotions, all_emotions
    )
    
    # Save metrics
    metrics_path = output_dir / "metrics_implicit.txt"
    save_metrics(
        accuracy_true, f1_macro_true, report_true, cm_true,
        accuracy_proxy, f1_macro_proxy, report_proxy, cm_proxy,
        all_emotions, metrics_path
    )
    print(f"Metrics saved to: {metrics_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"True vs Predicted - Accuracy: {accuracy_true:.4f} ({accuracy_true*100:.2f}%), F1 Macro: {f1_macro_true:.4f}")
    print(f"Proxy vs Predicted - Accuracy: {accuracy_proxy:.4f} ({accuracy_proxy*100:.2f}%), F1 Macro: {f1_macro_proxy:.4f}")
    print(f"\nAnalysis complete! Results saved in: {output_dir}")

    ## Now for NEUTRAL emotions
    # Extract emotions
    true_emotions, proxy_emotions, predicted_emotions = extract_emotions(df_other)
    
    # Get unique labels
    all_emotions = list(set(true_emotions + proxy_emotions + predicted_emotions))
    all_emotions.sort()  # Sort for consistent ordering
    
    # Create output directory
    csv_filename = Path(csv_filepath).stem
    output_dir = Path(output_path) / csv_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create confusion matrix for True vs Predicted
    cm_true_path = output_dir / "other_confusion_matrix_true_vs_predicted.png"
    cm_true = create_confusion_matrix_plot(
        true_emotions, predicted_emotions, all_emotions, cm_true_path,
        'Confusion Matrix: True Emotion vs Predicted Emotion',
        'Predicted Emotion',
        'True Emotion (from filename)'
    )
    print(f"True vs Predicted confusion matrix saved to: {cm_true_path}")
    
    # Create confusion matrix for Proxy vs Predicted
    cm_proxy_path = output_dir / "other_confusion_matrix_proxy_vs_predicted.png"
    cm_proxy = create_confusion_matrix_plot(
        proxy_emotions, predicted_emotions, all_emotions, cm_proxy_path,
        'Confusion Matrix: Proxy Emotion vs Predicted Emotion',
        'Predicted Emotion',
        'Proxy Emotion (from filename)'
    )
    print(f"Proxy vs Predicted confusion matrix saved to: {cm_proxy_path}")
    
    # Calculate metrics for True vs Predicted
    accuracy_true, f1_macro_true, report_true = calculate_metrics(
        true_emotions, predicted_emotions, all_emotions
    )
    
    # Calculate metrics for Proxy vs Predicted
    accuracy_proxy, f1_macro_proxy, report_proxy = calculate_metrics(
        proxy_emotions, predicted_emotions, all_emotions
    )
    
    # Save metrics
    metrics_path = output_dir / "metrics_other.txt"
    save_metrics(
        accuracy_true, f1_macro_true, report_true, cm_true,
        accuracy_proxy, f1_macro_proxy, report_proxy, cm_proxy,
        all_emotions, metrics_path
    )
    print(f"Metrics saved to: {metrics_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"True vs Predicted - Accuracy: {accuracy_true:.4f} ({accuracy_true*100:.2f}%), F1 Macro: {f1_macro_true:.4f}")
    print(f"Proxy vs Predicted - Accuracy: {accuracy_proxy:.4f} ({accuracy_proxy*100:.2f}%), F1 Macro: {f1_macro_proxy:.4f}")
    print(f"\nAnalysis complete! Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze emotion predictions from CSV file")
    parser.add_argument("csv_filepath", help="Path to the CSV file containing predictions")
    parser.add_argument("output_path", help="Path where output folder will be created")
    
    args = parser.parse_args()
    
    main(args.csv_filepath, args.output_path)