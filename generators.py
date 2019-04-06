from config import DATA_FRAME


class BaseGenerator:
    def __init__(self):
        raise NotImplementedError()

    def get_next(self):
        raise NotImplementedError()


class CSVGenerator(BaseGenerator):
    DATA_CHUNKS = []
    df = DATA_FRAME
    grouped_by_ts = df.groupby(["time_seen"])
    for group in grouped_by_ts.groups:
        DATA_CHUNKS.append(grouped_by_ts.get_group(group))

    def __init__(self):
        self.get_next_counter = 0
        self.chunks = self.DATA_CHUNKS

    def get_next(self):
        next_chunk = self.chunks[self.get_next_counter % len(self.chunks)]
        self.get_next_counter += 1
        return next_chunk
