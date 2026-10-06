import numpy as np

def compute_metrics(events: list) -> dict:
    if not events:
        return {
            'total_count': 0,
            'mean_val': 0.0,
            'max_val': None
        }
    total_count = len(events)
    values = [event[2] for event in events]
    mean_val = np.mean(values)
    max_val = max(values) if values else None
    return {
        'total_count': total_count,
        'mean_val': mean_val,
        'max_val': max_val
    }