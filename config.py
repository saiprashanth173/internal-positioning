import os

import pandas as pd
import glob

DATA_DIR = "data/ble_csv/ble-rssi-dataset"
GENERATOR = 'CSVGenerator'
DATA_FRAME = pd.concat([pd.read_csv(f, encoding='latin1') for f in glob.glob(DATA_DIR + '/*.csv')],
                       ignore_index=True)

MODEL_PATH = "data/rssi.hd5"

MULTI_DATA_DIR = "data/uji_data"
MULTI_GENERATOR = "CSVMultiGenerator"
MULTI_DATA_FRAME = pd.concat([pd.read_csv(f, encoding='latin1') for f in glob.glob(MULTI_DATA_DIR + '/*.csv')])
PREDICTION_MODEL = 'RandomForest'

PG_GENERATOR = "PGGenerator"


def raise_error(variable):
    raise EnvironmentError('The "' + variable + '" configuration must be defined')


class CeleryConfig:
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL') or raise_error('CELERY_BROKER_URL')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND') or raise_error('CELERY_RESULT_BACKEND')
    CELERY_IMPORTS = ('indoor_positioning.tasks',)
