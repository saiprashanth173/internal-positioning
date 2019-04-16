from config import MULTI_DATA_FRAME
from celery_worker import app
from tasks import predict_for
from gevent import sleep
import time
import pandas as pd


def uji_producer():
    DATA_CHUNKS = []
    df = MULTI_DATA_FRAME
    grouped_by_ts = df.groupby(["TIMESTAMP"])
    for group in grouped_by_ts.groups:
        DATA_CHUNKS.append(grouped_by_ts.get_group(group))
    i = 0
    while True:
        sleep(1)
        chunk = DATA_CHUNKS[i % len(DATA_CHUNKS)]
        predict_for.apply_async((chunk.to_json(orient='records'),))
        i += 1


if __name__ == '__main__':
    uji_producer()
    pass
