import random
import time

import pandas as pd
from keras.engine.saving import load_model

import train_lat_long_detector
from config import DATA_FRAME, MODEL_PATH, MULTI_DATA_FRAME, PREDICTION_MODEL
from db import get_latest_positions
from predict import predict


class BaseGenerator:
    def __init__(self):
        raise NotImplementedError()

    def get_next(self):
        raise NotImplementedError()


class CSVGenerator(BaseGenerator):
    DATA_CHUNKS = []
    df = DATA_FRAME
    model = load_model(MODEL_PATH)
    df["location"] = pd.DataFrame(predict(model, df.iloc[:, 2:]), index=df.index)
    df['date'] = pd.to_datetime(df.date)
    grouped_by_ts = df.groupby(["date"])
    for group in grouped_by_ts.groups:
        DATA_CHUNKS.append(grouped_by_ts.get_group(group))

    def __init__(self):
        self.get_next_counter = int(time.time())
        self.chunks = self.DATA_CHUNKS

    def get_next(self):
        next_chunk = self.chunks[self.get_next_counter % len(self.chunks)]
        self.get_next_counter += 1
        return next_chunk


class CSVMultiGenerator(BaseGenerator):
    DATA_CHUNKS = []
    model_class = getattr(train_lat_long_detector, PREDICTION_MODEL)
    model = model_class()
    df = MULTI_DATA_FRAME
    predictions = model.predict(train_lat_long_detector.get_x_y_from_df(df)[0])
    df[["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"]] = pd.DataFrame(predictions, index=df.index)
    df = df[["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID",
             "TIMESTAMP"]]
    grouped_by_ts = df.groupby(["TIMESTAMP"])
    for group in grouped_by_ts.groups:
        DATA_CHUNKS.append(grouped_by_ts.get_group(group))

    def __init__(self):
        random.seed(0)
        self.get_next_counter = random.randint(0, 100000)
        self.chunks = self.DATA_CHUNKS

    def get_next(self):
        next_chunk = self.chunks[self.get_next_counter % len(self.chunks)]
        self.get_next_counter += 1
        return next_chunk


class PGGenerator(BaseGenerator):

    def __init__(self):
        pass

    def get_next(self, user_id=None):
        return get_latest_positions(user_id)
