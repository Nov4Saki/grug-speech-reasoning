from typing import Optional
from datetime import datetime, timedelta

class TokenBucketLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.last_refill = 0
        self.current_tokens = 0

    def allow_request(self, tenant_id: str) -> bool:
        current_time = datetime.now()
        time_since_last_refill = current_time - self.last_refill
        tokens_needed = (time_since_last_refill.total_seconds() * self.refill_rate)
        tokens_available = self.current_tokens

        if tokens_available >= tokens_needed:
            self.current_tokens -= tokens_needed
            if self.current_tokens < 0:
                self.current_tokens = 0
            if self.current_tokens + tokens_needed > self.capacity:
                self.current_tokens = self.capacity
            return True
        else:
            self.current_tokens = 0
            if tokens_available + tokens_needed > self.capacity:
                self.current_tokens = self.capacity - tokens_needed
            else:
                self.current_tokens = 0
            self.last_refill = current_time
            return False