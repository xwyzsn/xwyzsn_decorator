# parallel_apply.py

import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cloudpickle

# -----------------------
# Helper: make any function picklable
# -----------------------
def make_picklable(func):
    serialized = cloudpickle.dumps(func)

    def wrapper(*args, **kwargs):
        real_func = cloudpickle.loads(serialized)
        return real_func(*args, **kwargs)

    wrapper.__name__ = getattr(func, '__name__', 'wrapped_lambda')
    return wrapper


# -----------------------
# Internal worker (must be global for pickling)
# -----------------------
def _apply_chunk(chunk, func_serialized, axis, raw, result_type, args):
    func = cloudpickle.loads(func_serialized)
    return chunk.apply(func, axis=axis, raw=raw, result_type=result_type, args=args)


# -----------------------
# Main function
# -----------------------
def parallel_apply(
    df: pd.DataFrame,
    func,
    axis=1,
    raw=False,
    result_type=None,
    args=(),
    is_parallel=False,
    chunks=None,
    show_progress=True,
):
    """
    Parallel-aware version of DataFrame.apply with progress bar and lambda support.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        func (callable): Function to apply (can be lambda, closure, or local)
        axis (int): 0 (columns) or 1 (rows)
        raw (bool): Whether to pass raw data
        result_type (str): expand / reduce / broadcast
        args (tuple): Additional args to func
        is_parallel (bool): Enable parallel execution
        chunks (int): Number of process chunks
        show_progress (bool): Whether to show progress bar

    Returns:
        pd.Series or pd.DataFrame
    """
    if not is_parallel:
        return df.apply(func, axis=axis, raw=raw, result_type=result_type, args=args)

    if chunks is None:
        chunks = multiprocessing.cpu_count()

    # Pickle the function (lambda or otherwise)
    func_serialized = cloudpickle.dumps(func)

    # Split data
    split_data = (
        np.array_split(df, chunks, axis=1 if axis == 0 else 0)
    )

    results = [None] * len(split_data)

    with ProcessPoolExecutor(max_workers=chunks) as executor:
        futures = {
            executor.submit(_apply_chunk, chunk, func_serialized, axis, raw, result_type, args): idx
            for idx, chunk in enumerate(split_data)
        }

        if show_progress:
            pbar = tqdm(total=len(futures), desc="Parallel apply")

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()

    return pd.concat(results)

if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        'A': range(10),
        'B': range(10, 20)
    })

    def example_func(row):
        return row['A'] + row['B']

    result = parallel_apply(df, example_func, axis=1, is_parallel=True, show_progress=True)
    print(result)
