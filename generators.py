from keras.engine.saving import load_model

from config import DATA_FRAME, MODEL_PATH
import pandas as pd

from predict import predict


class BaseGenerator:
    def __init__(self):
        raise NotImplementedError()

    def get_next(self):
        raise NotImplementedError()


# class CSVGenerator(BaseGenerator):
#     DATA_CHUNKS = []
#     df = DATA_FRAME
#     grouped_by_ts = df.groupby(["time_seen"])
#     for group in grouped_by_ts.groups:
#         DATA_CHUNKS.append(grouped_by_ts.get_group(group))

#     def __init__(self):
#         self.get_next_counter = 0
#         self.chunks = self.DATA_CHUNKS

#     def get_next(self):
#         next_chunk = self.chunks[self.get_next_counter % len(self.chunks)]
#         self.get_next_counter += 1
#         return next_chunk

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
        self.get_next_counter = 0
        self.chunks = self.DATA_CHUNKS

    def get_next(self):
        next_chunk = self.chunks[self.get_next_counter % len(self.chunks)]
        self.get_next_counter += 1
        return next_chunk
