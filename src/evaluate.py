import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': np.mean(np.array(y_true) == np.array(y_pred)),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
    }
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            )
        except ValueError:
            metrics['auc'] = 0.0
    return metrics


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['N', 'S', 'V', 'F', 'Q'],
                yticklabels=['N', 'S', 'V', 'F', 'Q'])
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tradeoff(results, save_path):
    """results: dict of {algo: {'f1': [...], 'comm': float}}"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'fedavg': 'blue', 'fedprox': 'green', 'scaffold': 'red'}
    for algo, data in results.items():
        ax.scatter(data['comm'], np.mean(data['f1']),
                   label=algo.upper(), color=colors.get(algo, 'gray'), s=100)
    ax.set_xlabel('Communication Cost (MB/round)')
    ax.set_ylabel('Macro F1-Score')
    ax.set_title('Performance vs. Communication Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()