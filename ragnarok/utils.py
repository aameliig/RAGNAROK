from ._libs import time, wraps


def measure_latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = (time.time() - start) * 1000
        return result, latency
    return wrapper
