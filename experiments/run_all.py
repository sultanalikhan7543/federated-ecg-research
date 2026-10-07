"""
Final federated ECG arrhythmia classification experiment.

Compares FedAvg and FedProx under non-IID Dirichlet client partitions
on the MIT-BIH Arrhythmia Database.

Produces:
    results/final_results.json
    results/final_summary.txt
    results/final_convergence.png
    results/final_f1_bars.png
    results/final_confusion_matrices.png

Usage:
    python experiments/run_all.py

Author: [Your Name]
Year: 2026
"""
import os
import sys
import json
import time

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'subsample': 0.20,        # 20% of training data (fast development)
    'test_subsample': 0.30,   # 30% of test data
    'rounds': 20,             # federated communication rounds
    'local_epochs': 2,        # local epochs per round
    'num_clients': 5,         # simulated hospital clients
    'alphas': [0.1, 0.5, 1.0],    # non-IID levels
    'algorithms': ['fedavg', 'fedprox'],  # SCAFFOLD dropped (unstable)
    'seed': 42,
    'batch_size': 128,
    'lr': 0.001,
    'proximal_mu': 0.01,
    'data_dir': 'data/',
    'results_dir': 'results',
}


# ============================================================
# DATA LOADING
# ============================================================
def load_data(data_dir='data/', subsample=1.0, test_subsample=1.0, seed=42):
    """Load MIT-BIH CSVs, normalize, and optionally subsample."""
    train_df = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), header=None)
    test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), header=None)

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)

    # Per-sample z-score normalization
    X_train = (X_train - X_train.mean(axis=1, keepdims=True)) / (
        X_train.std(axis=1, keepdims=True) + 1e-8)
    X_test = (X_test - X_test.mean(axis=1, keepdims=True)) / (
        X_test.std(axis=1, keepdims=True) + 1e-8)

    rng = np.random.default_rng(seed)
    if subsample < 1.0:
        n = int(subsample * len(X_train))
        idx = rng.choice(len(X_train), size=n, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if test_subsample < 1.0:
        n = int(test_subsample * len(X_test))
        idx = rng.choice(len(X_test), size=n, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    return X_train, y_train, X_test, y_test


def dirichlet_partition(y, num_clients, alpha, seed):
    """Partition class labels across clients using a Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    num_classes = int(y.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(np.repeat(alpha, num_clients))
        props = (props * len(idx)).astype(int)
        props[-1] = len(idx) - props[:-1].sum()
        start = 0
        for k in range(num_clients):
            client_indices[k].extend(idx[start:start + props[k]])
            start += props[k]

    # Guard against empty clients
    for k in range(num_clients):
        if len(client_indices[k]) == 0:
            client_indices[k] = list(rng.choice(len(y), size=50, replace=False))

    return client_indices


def compute_class_weights(y, num_classes=5):
    """Inverse-frequency weights to handle class imbalance."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# MODEL
# ============================================================
class FastECGNet(nn.Module):
    """Compact 1D CNN for 187-step ECG heartbeat classification."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)                # (B, 187) -> (B, 1, 187)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================
# FEDERATED CLIENT
# ============================================================
class Client:
    def __init__(self, cid, X, y, class_weights, batch_size=128,
                 lr=0.001, device='cpu'):
        self.cid = cid
        self.device = device
        self.loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.lr = lr
        self.class_weights = class_weights.to(device)
        self.n_samples = len(X)

    def train(self, model, epochs=2, proximal_mu=0.0, global_params=None):
        """
        Local training with class-weighted cross-entropy loss.

        If proximal_mu > 0 and global_params is provided, adds a FedProx
        proximal term ||w - w_global||^2.

        global_params is a dict {parameter_name: tensor} containing only
        trainable parameters (not BatchNorm buffers).
        """
        model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        model.train()

        for _ in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)

                # FedProx proximal term (only for trainable parameters)
                if proximal_mu > 0 and global_params is not None:
                    prox = torch.tensor(0.0, device=self.device)
                    for name, p in model.named_parameters():
                        if name in global_params:
                            gp = global_params[name].to(self.device)
                            prox = prox + ((p - gp) ** 2).sum()
                    loss = loss + (proximal_mu / 2.0) * prox

                loss.backward()
                opt.step()

        return {k: v.cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# AGGREGATION AND EVALUATION
# ============================================================
def fedavg_aggregate(client_weights, client_sizes):
    """Weighted average of client state dicts."""
    total = sum(client_sizes)
    new_state = {}
    for key in client_weights[0]:
        new_state[key] = sum(
            cw[key].float() * (sz / total)
            for cw, sz in zip(client_weights, client_sizes)
        )
    return new_state


def full_eval(model, X_test, y_test, device):
    """Evaluate model on the test set."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test).to(device))
        _, preds = torch.max(out, 1)
        preds = preds.cpu().numpy()
    return {
        'acc': float(accuracy_score(y_test, preds)),
        'f1': float(f1_score(y_test, preds, average='macro')),
        'preds': preds,
        'y_true': y_test,
    }


# ============================================================
# FEDERATED TRAINING LOOP
# ============================================================
def run_federated(clients, algo, rounds, local_epochs, device,
                  prox_mu, lr, test_data, verbose=True):
    """Run one federated learning experiment for a given algorithm."""
    model = FastECGNet().to(device)
    global_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    history = {'round': [], 'acc': [], 'f1': []}

    param_names = set(dict(model.named_parameters()).keys())

    it = tqdm(range(1, rounds + 1), desc=f"  {algo:>8}",
              leave=False, disable=not verbose)

    for r in it:
        client_weights, client_sizes = [], []

        # Only include trainable params (skip BatchNorm buffers)
        global_params_dict = {
            k: v for k, v in global_state.items() if k in param_names
        }

        for client in clients:
            local_model = FastECGNet().to(device)
            local_model.load_state_dict(global_state)

            if algo == 'fedprox':
                w = client.train(
                    local_model, epochs=local_epochs,
                    proximal_mu=prox_mu,
                    global_params=global_params_dict
                )
            else:  # fedavg
                w = client.train(local_model, epochs=local_epochs)

            client_weights.append(w)
            client_sizes.append(client.n_samples)

        # Server aggregation
        global_state = fedavg_aggregate(client_weights, client_sizes)
        model.load_state_dict(global_state)

        # Evaluate
        m = full_eval(model, test_data[0], test_data[1], device)
        history['round'].append(r)
        history['acc'].append(m['acc'])
        history['f1'].append(m['f1'])
        it.set_postfix(acc=f"{m['acc']:.3f}", f1=f"{m['f1']:.3f}")

    return model, history


# ============================================================
# PLOTTING
# ============================================================
def make_convergence_plot(results, alphas, algorithms, save_path):
    plt.figure(figsize=(9, 5.5))
    for algo in algorithms:
        for a in alphas:
            h = results[a][algo]['history']
            plt.plot(h['round'], h['f1'], marker='o', ms=3,
                     label=f"{algo.upper()} (α={a})")
    plt.xlabel('Federated Round')
    plt.ylabel('Macro F1-Score')
    plt.title('Convergence Curves')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def make_f1_bars(results, alphas, algorithms, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(alphas))
    width = 0.35
    for i, algo in enumerate(algorithms):
        f1s = [results[a][algo]['f1'] for a in alphas]
        ax.bar(x + i * width, f1s, width, label=algo.upper())
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_ylabel('Macro F1-Score')
    ax.set_title('F1-Score by Algorithm and Non-IID Level')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def make_confusion_matrices(results, alpha_focus, algorithms, save_path):
    fig, axes = plt.subplots(1, len(algorithms),
                             figsize=(5.5 * len(algorithms), 4.5))
    if len(algorithms) == 1:
        axes = [axes]
    for ax, algo in zip(axes, algorithms):
        r = results[alpha_focus][algo]
        cm = confusion_matrix(r['y_true'], r['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['N', 'S', 'V', 'F', 'Q'],
                    yticklabels=['N', 'S', 'V', 'F', 'Q'])
        ax.set_title(f"{algo.upper()} (α={alpha_focus})\nF1={r['f1']:.3f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()

    print("=" * 65)
    print("FINAL Federated ECG Experiment")
    print("=" * 65)
    print(f"Device:         {device}")
    print(f"Subsample:      {CONFIG['subsample'] * 100:.0f}%")
    print(f"Rounds:         {CONFIG['rounds']}")
    print(f"Local epochs:   {CONFIG['local_epochs']}")
    print(f"Clients:        {CONFIG['num_clients']}")
    print(f"Alphas:         {CONFIG['alphas']}")
    print(f"Algorithms:     {CONFIG['algorithms']}")
    print("=" * 65)

    # ---- Load data ----
    print("\n[1/4] Loading data...")
    X_train, y_train, X_test, y_test = load_data(
        data_dir=CONFIG['data_dir'],
        subsample=CONFIG['subsample'],
        test_subsample=CONFIG['test_subsample'],
        seed=CONFIG['seed'],
    )
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"      Class counts (train): {np.bincount(y_train, minlength=5)}")
    class_weights = compute_class_weights(y_train, num_classes=5)
    print(f"      Class weights: {class_weights.numpy().round(3)}")

    # ---- Run experiments ----
    print("\n[2/4] Running federated experiments...")
    results = {}
    for alpha in CONFIG['alphas']:
        results[alpha] = {}
        idx_part = dirichlet_partition(
            y_train, CONFIG['num_clients'], alpha, CONFIG['seed']
        )
        clients = [
            Client(i, X_train[idx_part[i]], y_train[idx_part[i]], class_weights,
                   batch_size=CONFIG['batch_size'], lr=CONFIG['lr'],
                   device=device)
            for i in range(CONFIG['num_clients'])
        ]

        for algo in CONFIG['algorithms']:
            print(f"\n  α={alpha} | {algo}")
            model, hist = run_federated(
                clients, algo,
                rounds=CONFIG['rounds'],
                local_epochs=CONFIG['local_epochs'],
                device=device,
                prox_mu=CONFIG['proximal_mu'],
                lr=CONFIG['lr'],
                test_data=(X_test, y_test),
                verbose=True,
            )
            m = full_eval(model, X_test, y_test, device)
            results[alpha][algo] = {
                'f1': m['f1'],
                'acc': m['acc'],
                'history': hist,
                'preds': m['preds'],
                'y_true': m['y_true'],
            }
            print(f"     → F1: {m['f1']:.4f} | Acc: {m['acc']:.4f}")
            print(f"     → Pred dist: {np.bincount(m['preds'], minlength=5)}")

    # ---- Save results (with UTF-8 and JSON-safe types) ----
    print("\n[3/4] Saving results...")

    serializable = {
        str(a): {
            algo: {
                'f1': float(results[a][algo]['f1']),
                'acc': float(results[a][algo]['acc']),
                'history': {
                    'round': [int(x) for x in results[a][algo]['history']['round']],
                    'acc':   [float(x) for x in results[a][algo]['history']['acc']],
                    'f1':    [float(x) for x in results[a][algo]['history']['f1']],
                },
                'pred_dist': [
                    int(x)
                    for x in np.bincount(results[a][algo]['preds'], minlength=5)
                ],
            }
            for algo in results[a]
        }
        for a in results
    }

    json_path = os.path.join(CONFIG['results_dir'], 'final_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)

    # Summary table
    lines = [
        "=" * 65,
        "FINAL EXPERIMENT SUMMARY",
        "=" * 65,
        f"{'Algorithm':<12}{'α':<8}{'F1':<12}{'Accuracy':<12}",
        "-" * 65,
    ]
    for a in CONFIG['alphas']:
        for algo in CONFIG['algorithms']:
            r = results[a][algo]
            lines.append(
                f"{algo:<12}{a:<8}{r['f1']:.4f}      {r['acc']:.4f}"
            )
    lines.append("-" * 65)
    lines.append(f"Total time: {time.time() - t0:.1f} sec")
    summary = "\n".join(lines)

    summary_path = os.path.join(CONFIG['results_dir'], 'final_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(summary)

    # ---- Generate figures ----
    print("\n[4/4] Generating figures...")

    make_convergence_plot(
        results, CONFIG['alphas'], CONFIG['algorithms'],
        os.path.join(CONFIG['results_dir'], 'final_convergence.png')
    )

    make_f1_bars(
        results, CONFIG['alphas'], CONFIG['algorithms'],
        os.path.join(CONFIG['results_dir'], 'final_f1_bars.png')
    )

    make_confusion_matrices(
        results, CONFIG['alphas'][0], CONFIG['algorithms'],
        os.path.join(CONFIG['results_dir'], 'final_confusion_matrices.png')
    )

    print("\n" + "=" * 65)
    print("DONE — files written to results/:")
    print("  • final_results.json")
    print("  • final_summary.txt")
    print("  • final_convergence.png")
    print("  • final_f1_bars.png")
    print("  • final_confusion_matrices.png")
    print(f"\nTotal time: {time.time() - t0:.1f} sec")
    print("=" * 65)


if __name__ == '__main__':
    main()