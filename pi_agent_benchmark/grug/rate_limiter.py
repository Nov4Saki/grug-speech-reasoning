import time

class TokenBucketLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.last requesting_time = 0
        self.current_token_count = 0

    def allow_request(self, tenant_id: str) -> bool:
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        tokens_used = time_since_last * self.refill_rate
        self.current_token_count -= tokens_used
        if self.current_token_count < 0:
            self.current_token_count = 0
        self.last_request_time = current_time
        return self.current_token_count >= 0 and self.current_token_count >= self.capacity