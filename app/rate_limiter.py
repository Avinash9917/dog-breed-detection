import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from app.config import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS

class InMemoryRateLimiter:
    """
    Sliding window in-memory rate limiter keyed by client IP.
    Thread-safe for typical asynchronous FastAPI event loop.
    """
    def __init__(self, max_requests: int = RATE_LIMIT_MAX_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_history = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Clean older timestamps
        timestamps = [t for t in self.request_history[client_ip] if t > cutoff]
        
        if len(timestamps) >= self.max_requests:
            self.request_history[client_ip] = timestamps
            return False
            
        timestamps.append(now)
        self.request_history[client_ip] = timestamps
        return True

    def reset(self) -> None:
        """Clear all rate limit history (useful for test isolation)."""
        self.request_history.clear()


limiter = InMemoryRateLimiter()


def rate_limit_dependency(request: Request):
    """
    FastAPI dependency to enforce client rate limiting.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    
    # Handle X-Forwarded-For if behind reverse proxy
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    if not limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before submitting more requests."
        )
