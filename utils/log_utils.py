from functools import wraps


def time_log(func):
    @wraps(func)
    def wrap(func):
        return func

    return func
