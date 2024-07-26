

def retry(tries,exception_exclude_list=[],vvvv=False):
    """
    Decorator that retries a function a number of times before giving up.
    :param tries: number of tries before giving up
    :param exception_exclude_list: list of exceptions to exclude from retry
    :param vvvv: print verbose output
    """
    def _retry(func):
        def wrapper(*args, **kwargs):
            for i in range(tries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if vvvv:
                        print(f"{func.__name__} try {i} error, {e}")
                    if type(e) in exception_exclude_list or any([isinstance(e, exception) for exception in exception_exclude_list]):
                        raise e
                    else:
                        continue
        return wrapper
    return _retry
