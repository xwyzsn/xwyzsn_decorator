# Decorator 

some decorators for personal usage.

## install

```bash
pip install xwyzsn-decorator -i https://pypi.python.org/simple
```

## Usage

### timer

```python
import time 
import random 
from x_decorator.timer import timer

@timer(vvvv=True)
def idle_fn():
    time.sleep(random.randint(0,5))
    return "finish"


if __name__ == "__main__":
    idle_fn()
    print(idle_fn.total_time)

# dle_fn started at 1722061281.072323
# idle_fn ended at 1722061283.07743
# idle_fn took 2.0051069259643555 seconds
# 2.0051069259643555

```

### retry

```python

import random
from x_decorator.retry import retry

# tries: number of attempts
# exception_exclude_list: [Exceptions that you dont want to retry ]

@retry(tries=3,exception_exclude_list=[OSError],
       vvvv=True)
def fn_may_fail():
    if random.randint(0, 1) == 1:
        print("Success")
        return "Success"
    else:
        print("Fail")
        raise ValueError("Fail")
        
    
if __name__ == "__main__":
    fn_may_fail()

```

### search_with_cuda

```python
from x_decorator.search import search_with_cuda
import random
import time 
import os 


if __name__ == "__main__":
    """
    search_space: Dict[str,list], the final search space will be the product of values
    workers: int, number of conconruent process.
    call_back_fn: callable, clean up function when all process is done.
    devices: available devices, e.g., [1,1,1] means there are at most three process can run on device `1`. 
    db_path: str, where to store the sqlite3 database.
    gradio_fn: callable, how to fetch data from sqlite3. by default, 
    
    ```python
        def fetch_data():
            df = pd.DataFrame(_read(db_path))
            if len(df) == 0:
                return pd.DataFrame([])
            df.columns = ['id', 'config', 'result']
            result = pd.concat([json_normalize(df['config']), json_normalize(df['result'])], axis=1)
            return result
    ``` 

    """


    @search_with_cuda(search_space={"dim": [64, 128, 256, 512], "dim2": [512, 720]}, workers=4,
    db_path='/path/to/[xxx].db',
    call_back_fn=lambda x: print(x), devices=["1", "2"],
    gradio_fn= lambda x:x
    )
    def idle_fn(config):
        sleep = random.randint(1, 10)
        #  you can access config['dim'], config['dim2'] ,etc.
        # training 
        print(f"{os.getpid()} ENV{os.getenv('test', None)} sleep")
        time.sleep(sleep)

        #  return your eval result 
        # result need to be a dict 
        return {"pid": os.getpid(), "config": config, "sleep": sleep}
    results = idle_fn() 
    print(results)

```

### parallel_apply

```python
from x_decorator.parallel_apply import parallel_apply
import pandas as pd 
# pd.DataFrame.p_apply = parallel_apply
if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        'A': range(10),
        'B': range(10, 20)
    })
    def example_func(row):
        return row['A'] + row['B']
    # or df.p_apply(example_func,is_parallel=True, show_progress=True)
    result = parallel_apply(df, example_func, axis=1, is_parallel=True, show_progress=True)
    print(result)

```