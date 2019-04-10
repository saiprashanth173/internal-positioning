import json
from pprint import pprint

import train_lat_long_detector
from celery_worker import app
import time
import pandas as pd

from config import PREDICTION_MODEL
from db import insert_positions


@app.task
def longtime_add(x, y):
    print('long time task begins')
    # sleep 5 seconds
    time.sleep(5)
    print('long time task finished')
    return x + y


@app.task
def predict_for(df):
    model_class = getattr(train_lat_long_detector, PREDICTION_MODEL)
    model = model_class()
    # print()
    df = pd.DataFrame(json.loads(df))
    predictions = model.predict(train_lat_long_detector.get_x_y_from_df(df)[0])
    df[["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"]] = pd.DataFrame(predictions, index=df.index)
    df = df[["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID",
             "TIMESTAMP"]]
    insert_positions(df.to_json(orient='records'))
    pprint(df)
