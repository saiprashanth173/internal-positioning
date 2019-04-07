import random
import sys
import pandas as pd
from train import fix_pos
from keras.models import load_model
from pprint import pprint


def rev_pos(predicts):
    vals = []
    for pred in predicts:
        converted = 87 - int(round(pred[0]))
        if converted < ord('A') or converted > ord('Z'):
            converted = random.randint(ord('A'), ord('Z'))
        pred_ = [chr(converted), int(round(pred[1]))]
        vals.append(pred_)
    return vals


def predict(model, X):
    return model.predict(X)


def predict_all(csv_file, model_path="data/rssi.hd5"):
    df = pd.read_csv(csv_file, index_col=None)
    df['x'] = df['location'].str[0]
    df['y'] = df['location'].str[1:]
    df.drop(["location"], axis=1, inplace=True)
    df["x"] = df["x"].apply(fix_pos)
    X = df.iloc[:, 1:-2]
    model = load_model(model_path)
    return predict(model, X)


if __name__ == "__main__":
    pprint(rev_pos(predict_all(sys.argv[1])))
