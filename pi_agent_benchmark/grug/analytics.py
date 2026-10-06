import sqlite3

def compute_metrics(events: list) -> dict:
    if not events:
        return {'total_count': 0, 'mean_val': 0.0, 'max_val': None}
    
    total_count = len(events)
    sum_val = sum(event['val'] for event in events)
    max_val = max(event['val'] for event in events) if events else None
    mean_val = sum_val / total_count if total_count > 0 else None
    
    return {
        'total_count': total_count,
        'mean_val': mean_val,
        'max_val': max_val
    }