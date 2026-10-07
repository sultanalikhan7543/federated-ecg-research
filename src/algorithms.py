import torch
import copy

def fedavg_aggregate(global_model, client_weights, client_sizes):
    """Weighted average of client state dicts."""
    total = sum(client_sizes)
    new_state = copy.deepcopy(global_model.state_dict())
    for key in new_state:
        new_state[key] = sum(
            client_weights[k][key] * (client_sizes[k] / total)
            for k in range(len(client_weights))
        )
    return new_state


def scaffold_aggregate(global_model, client_weights, client_cvs, server_cv, client_sizes, lr):
    """SCAFFOLD aggregation: model + control variates."""
    total = sum(client_sizes)
    new_state = copy.deepcopy(global_model.state_dict())
    new_cv = copy.deepcopy(server_cv)

    for key in new_state:
        # Aggregate models
        new_state[key] = sum(
            client_weights[k][key] * (client_sizes[k] / total)
            for k in range(len(client_weights))
        )
        # Update server control variate
        new_cv[key] = server_cv[key] + (1.0 / len(client_weights)) * sum(
            client_cvs[k][key] - server_cv[key] for k in range(len(client_cvs))
        )
    return new_state, new_cv