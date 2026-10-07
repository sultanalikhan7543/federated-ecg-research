import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_mitbih(data_dir='data/'):
    """Load MIT-BIH preprocessed CSV files."""
    train_df = pd.read_csv(f'{data_dir}mitbih_train.csv', header=None)
    test_df = pd.read_csv(f'{data_dir}mitbih_test.csv', header=None)

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)

    # Z-score normalization per sample
    mean = X_train.mean(axis=1, keepdims=True)
    std = X_train.std(axis=1, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - X_test.mean(axis=1, keepdims=True)) / (X_test.std(axis=1, keepdims=True) + 1e-8)

    return X_train, y_train, X_test, y_test


def dirichlet_partition(y, num_clients=5, alpha=0.5, seed=42):
    """Partition labels into clients using Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(y))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(y == c)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(class_indices)).astype(int)
        # Adjust for rounding
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        start = 0
        for k in range(num_clients):
            client_indices[k].extend(class_indices[start:start + proportions[k]])
            start += proportions[k]

    return client_indices


def create_federated_data(X_train, y_train, num_clients=5, alpha=0.5, seed=42):
    """Return list of (X_client, y_client) tuples."""
    indices = dirichlet_partition(y_train, num_clients, alpha, seed)
    clients = []
    for idx in indices:
        clients.append((X_train[idx], y_train[idx]))
    return clients