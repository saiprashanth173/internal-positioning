import pandas as pd
import glob
from predict import predict
import numpy as np  # linear algebra
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from pandas import read_csv
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

from ble import *
from train import l2_dist, create_deep, fix_pos



from numpy.random import seed
seed(7)

def get_data():
    # Load dataset
    df = pd.concat([pd.read_csv(f, encoding='latin1') for f in glob.glob("data/ble_csv/ble-rssi-dataset/*.csv1")],
                        ignore_index=True)
    df['date'] = pd.to_datetime(df.date)

    df['x'] = df['location'].str[0]
    df['y'] = df['location'].str[1:]
    df.drop(["location"], axis=1, inplace=True)

    df["x"] = df["x"].apply(fix_pos)
    df["y"] = df["y"].astype(int)

    y = df.iloc[:, -2:]
    x = df.iloc[:, 1:-2]
    return train_test_split(x, y, test_size=.2, shuffle=True, random_state=5)

def dump_results():
    train_x, val_x, train_y, val_y = get_data()
    es = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto', restore_best_weights=True)
    model = create_deep(train_x.shape[1])
    
    history = model.fit(x=train_x,
              y=train_y,
              validation_data=(val_x, val_y),
              epochs=1000,
              batch_size=1000,
              verbose=1,
              callbacks=[es])


    preds = model.predict(val_x)
    l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))
    print("Model Error: ", l2dists_mean)

    with open('./easy_train_history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def find_max_rssi(row):
    max_rssi = float("-inf")
    max_index = None
    for i in range(1, 14):
        rssi = row['b30'+str(i).zfill(2)]
        if rssi > max_rssi:
            max_rssi = rssi
            max_index = i
    return max_index-1

def generate_graph():
    file_pi = open('./easy_train_history.pkl', 'rb')
    model_history = pickle.load(file_pi)

    print(model_history.keys())


############################
    train_x, val_x, train_y, val_y = get_data()
    points = get_points()
    # Get predictions for validation data
    val_tri_x_coord = []
    val_tri_y_coord = []
    for _, row in val_x.iterrows():
        circle_list = []
        for i in range(1, 14):
            rssi = row['b30'+str(i).zfill(2)]
            if rssi != -200:
                c = circle(points[i-1], calculate_dist_from_rssi(rssi))
                circle_list.append(c)
        if len(circle_list) < 3:
            center = points[find_max_rssi(row)]
        else:
            center = get_center(circle_list)
        val_tri_x_coord.append(int(round(center.x)))
        val_tri_y_coord.append(int(round(center.y)))

    val_tri_preds = pd.DataFrame({'x': val_tri_x_coord, 'y': val_tri_y_coord}, index=val_y.index)
    l2dists_mean, l2dists = l2_dist((val_tri_preds['x'], val_tri_preds['y']), (val_y["x"], val_y["y"]))
    print("Triliteration Error: ", l2dists_mean)

##############################
    colormap = mpl.cm.Dark2.colors

    plt.plot(model_history['acc'], c=colormap[0])
    plt.plot(model_history['val_acc'], c=colormap[1])
    plt.title('Easy Case: Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Cross Validation'])
    plt.grid()
    plt.savefig('Easy_Acc.png', dpi=150)

    plt.figure()

    epochs = len(model_history['loss'])
    plt.plot(model_history['loss'], c=colormap[0])
    plt.plot(model_history['val_loss'], c=colormap[1])
    plt.plot([l2dists_mean for i in range(1, epochs+1)], c=colormap[2])
    plt.title('Easy Case: Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Cross Validation', 'Triliteration'])
    plt.grid()
    plt.savefig('Easy_Loss.png', dpi=150)

if __name__ == "__main__":
    # dump_results()
    generate_graph()