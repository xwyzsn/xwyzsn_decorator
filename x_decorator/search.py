import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from functools import wraps, partial
from itertools import product
from tornado import concurrent
from multiprocessing import Process
import cloudpickle
from typing import Dict, List, Any

import sqlite3
import pandas as pd
from pandas import json_normalize
import json
import gradio as gr
import hashlib
import traceback

sqlite3.register_adapter(dict, json.dumps)
sqlite3.register_adapter(list, json.dumps)
sqlite3.register_converter("JSON", json.loads)


def dict_hash(dictionary: Dict[str, Any]) -> str:
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def apply_cloudpickle(fn, /, *args, **kwargs):
    fn = cloudpickle.loads(fn)
    return fn(*args, **kwargs)


class CloudpickleProcessPoolExecutor(ProcessPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(apply_cloudpickle, cloudpickle.dumps(fn), *args, **kwargs)


def create_db(db_path: str):
    try:
        create_table = """
            CREATE TABLE IF NOT EXISTS result (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                config JSON,
                `result` JSON
                )
            """
        conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        cursor.execute(create_table)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        raise e.with_traceback(e.__traceback__)


def _write(db_path: str, results: list[dict], write_list: list):
    try:
        conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        newly_inserted = []
        for result in list(results):
            if result not in list(write_list):
                newly_inserted.append(result)
        config = [r["config"] for r in newly_inserted]
        result = [r["result"] for r in newly_inserted]
        for c, r in zip(config, result):
            cursor.execute("INSERT INTO result ( config, result) VALUES (?, ?) ", (c, r))
        conn.commit()
        write_list.extend(newly_inserted)
        return True
    except Exception as e:
        raise e.with_traceback(e.__traceback__)

def write_process(result: list[dict], write_list: list, lock):
    try:
        time.sleep(10)
        with lock:
            _write(result, write_list)
    except Exception as e:
        raise e.with_traceback(e.__traceback__)

def _read(df_path):
    conn = sqlite3.connect(df_path, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    result = c.execute("SELECT * FROM result").fetchall()
    conn.close()
    return result


def gradio_process(db_path, fn=None):
    def fetch_data():
        df = pd.DataFrame(_read(db_path))
        if len(df) == 0:
            return pd.DataFrame([])
        df.columns = ['id', 'config', 'result']
        result = pd.concat([json_normalize(df['config']), json_normalize(df['result'])], axis=1)
        return result

    demo = gr.Interface(None, [], outputs=[
        gr.DataFrame(value=[fetch_data, fn][fn is not None], every=10, height=300, interactive=True)],
                        live=True)
    demo.launch()


def search_with_cuda(search_space: dict, db_path: str, workers: int = 2, call_back_fn=lambda x: x, devices=None,
                     gradio_fn=None):
    def wrapper(func):
        create_db(db_path)
        space = search_space.values()
        search_item = list(product(*space))
        keys = search_space.keys()
        search_item = [dict(list(zip(keys, item))) for item in search_item]
        manager = multiprocessing.Manager()
        q = manager.Queue()
        semaphore = manager.Semaphore(len(devices))
        results = manager.list()
        write_list = manager.list()
        gr_process = Process(target=gradio_process, args=(db_path, gradio_fn))
        gr_process.start()
        lock = manager.Lock()
        for item in devices:
            q.put(item)
        def fn(func, config, semaphore, q, results, lock, *args, **kwargs):
            semaphore.acquire()
            device = q.get()
            config["device"] = [int(device)]
            func_fn = partial(func, config)
            try:
                result = func_fn(*args, **kwargs)
            except Exception as e:
                e.with_traceback(e.__traceback__)
            with lock:
                results.append({"config": config, "result": result})
                _write(db_path, results, write_list)
            q.put(device)
            semaphore.release()
        @wraps(func)
        def inner(*args, **kwargs):
            try:
                with CloudpickleProcessPoolExecutor(max_workers=workers) as executor:
                    jobs = [executor.submit(partial(fn, func, config, semaphore, q, results, lock), *args, **kwargs)
                            for config in search_item]
                    for job in concurrent.futures.as_completed(jobs):
                        job.result()
                    _write(db_path, results, write_list)
                    return call_back_fn(list(results))
            except Exception as e:
                call_back_fn(list(results))
                e.with_traceback(e.__traceback__)

        return inner
    return wrapper


if __name__ == '__main__':
    @search_with_cuda(search_space={"dim": [64, 128, 256, 512], "look_back": [512, 720]}, workers=4,
                      call_back_fn=lambda x: print(x), devices=["1", "2"])
    def idle_fn(config):
        sleep = random.randint(1, 10)
        print(f"{os.getpid()} ENV{os.getenv('test', None)} sleep")
        time.sleep(sleep)

        return {"pid": os.getpid(), "config": config, "sleep": sleep}
    results = idle_fn()
    print(results)
