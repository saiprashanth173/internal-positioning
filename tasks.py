import json
import time
from pprint import pprint

import pandas as pd

import train_lat_long_detector
from celery_worker import app
from config import PREDICTION_MODEL
from db import insert_positions


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
