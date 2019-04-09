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
