import time 

def timer(vvvv=False):
    """
    A decorator that times a function
    :param vvvv: print verbose output
    """
    def warpper(func):
        def inner(*args, **kwargs):
            start = time.time()
            if vvvv:
                print(f"{func.__name__} started at {start}")
            result = func(*args, **kwargs)
            end = time.time()
            inner.total_time = end - start
            if vvvv:
                print(f"{func.__name__} ended at {end}")
                print(f"{func.__name__} took {end - start} seconds")
            return result
        inner.total_time = 0
        return inner
    return warpper


if __name__ == "__main__":
    @timer()
    def test():
        time.sleep(2)
        return "done"
    print(test())
    print(test.total_time)