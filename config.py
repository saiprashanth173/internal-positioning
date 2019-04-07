import pandas as pd
import glob

DATA_DIR = "data/ble_csv/ble-rssi-dataset"
GENERATOR = 'CSVGenerator'
DATA_FRAME = pd.concat([pd.read_csv(f, encoding='latin1') for f in glob.glob(DATA_DIR + '/*.csv')],
                       ignore_index=True)