import os
import sys
import time
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import check_rate_limit, rate_limit_store, settings


def test_rate_limit_enforcement_and_cleanup():
    client_id = "test-client"

    # Ensure isolation before test
    rate_limit_store.clear()
    assert client_id not in rate_limit_store

    # Allow exactly `rate_limit_requests` calls
    for _ in range(settings.rate_limit_requests):
        assert check_rate_limit(client_id) is True

    # Store should track all calls within the window
    assert len(rate_limit_store[client_id]) == settings.rate_limit_requests

    # Next call should be rejected
    assert check_rate_limit(client_id) is False
    assert len(rate_limit_store[client_id]) == settings.rate_limit_requests

    # Simulate passing of the rate limit window to trigger cleanup
    expired = time.time() - settings.rate_limit_window - 1
    rate_limit_store[client_id] = deque(
        [expired] * settings.rate_limit_requests,
        maxlen=settings.rate_limit_requests,
    )
    assert check_rate_limit(client_id) is True
    assert len(rate_limit_store[client_id]) == 1

    # Ensure isolation after test
    rate_limit_store.clear()
    assert client_id not in rate_limit_store

