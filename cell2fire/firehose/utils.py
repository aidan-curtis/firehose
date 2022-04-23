import argparse
import json
import os
import random
import string
import time


class ArgparseEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, argparse.Namespace):
            return obj.__dict__
        return super().default(obj)


def wait_until_file_populated(
    f_path: str, timeout: int = 10, min_size: int = 100, wait_time: float = 0.01
):
    """
    Wait until the file is populated with data.
    """
    start_time = time.perf_counter()

    while not os.path.exists(f_path) or os.stat(f_path).st_size < min_size:
        if time.perf_counter() - start_time > timeout:
            raise RuntimeError(f"Timed out waiting for {f_path} to be populated.")
        time.sleep(wait_time)


def random_string(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))
