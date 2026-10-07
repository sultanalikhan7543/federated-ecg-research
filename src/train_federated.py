import torch
import numpy as np
from tqdm import tqdm
from src.model import ECG1DCNN
from src.client import FLClient
from src.algorithms import fedavg_aggregate, scaffold_aggregate


def train_federated(
    clients,
    algorithm='fedavg',
    rounds=50,
    local_epochs=3,
    proximal_mu=0.01,
    lr=0.001,
    device='cpu',
    verbose=True
):
    """
    Main federated training loop.

    Args:
        clients: list of FLClient objects
        algorithm: 'fedavg' | 'fedprox' | 'scaffold'
        rounds: number of federated communication rounds
        local_epochs: number of local training epochs per round
        proximal_mu: proximal term coefficient (only for fedprox)
        lr: learning rate (used by SCAFFOLD)
        device: 'cpu' or 'cuda'
        verbose: print progress

    Returns:
        global_model: trained global model
        history: dict with 'round' and 'accuracy' lists
    """
    global_model = ECG1DCNN(num_classes=5).to(device)
    global_state = global_model.state_dict()

    # Initialize SCAFFOLD control variates if needed
    if algorithm == 'scaffold':
        server_cv = {k: torch.zeros_like(v) for k, v in global_state.items()}
        client_cvs = [
            {k: torch.zeros_like(v) for k, v in global_state.items()}
            for _ in clients
        ]

    history = {'round': [], 'loss': [], 'accuracy': []}

    # Progress bar over rounds
    round_iter = tqdm(
        range(1, rounds + 1),
        desc=f"  {algorithm:>8}",
        leave=False,
        disable=not verbose
    )

    for r in round_iter:
        client_weights = []
        client_sizes = []

        # ---- Local training on each client ----
        for client in clients:
            local_model = ECG1DCNN(num_classes=5).to(device)
            local_model.load_state_dict(global_state)

            if algorithm == 'fedprox':
                w = client.train(
                    local_model,
                    epochs=local_epochs,
                    proximal_mu=proximal_mu,
                    global_weights=[v for v in global_state.values()]
                )
            elif algorithm == 'scaffold':
                w = client.train(
                    local_model,
                    epochs=local_epochs,
                    proximal_mu=0.0
                )
            else:  # fedavg
                w = client.train(local_model, epochs=local_epochs)

            client_weights.append(w)
            client_sizes.append(len(client.dataset))

        # ---- Server aggregation ----
        if algorithm == 'scaffold':
            global_state, server_cv = scaffold_aggregate(
                global_model,
                client_weights,
                client_cvs,
                server_cv,
                client_sizes,
                lr
            )
        else:
            global_state = fedavg_aggregate(
                global_model,
                client_weights,
                client_sizes
            )

        global_model.load_state_dict(global_state)

        # ---- Evaluate globally (weighted by client size) ----
        total_correct, total_samples = 0, 0
        for client in clients:
            acc, _, _ = client.evaluate(global_model)
            n = len(client.dataset)
            total_correct += acc * n
            total_samples += n
        avg_acc = total_correct / total_samples

        history['round'].append(r)
        history['accuracy'].append(avg_acc)

        # Live update in progress bar
        round_iter.set_postfix(acc=f"{avg_acc:.4f}")

        if verbose and r % 10 == 0:
            print(f"    Round {r:3d} | Global Acc: {avg_acc:.4f}")

    return global_model, history