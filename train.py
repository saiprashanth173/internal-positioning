import numpy as np  # linear algebra
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from pandas import read_csv
from sklearn.model_selection import train_test_split


def l2_dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    dx = x1 - x2
    dy = y1 - y2
    dx = dx ** 2
    dy = dy ** 2
    dists = dx + dy
    dists = np.sqrt(dists)
    return np.mean(dists), dists


def create_deep(inp_dim):
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(50, input_dim=inp_dim, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='relu'))
    # Compile model
    model.compile(loss='mse', optimizer=Adam(.001), metrics=['mse'])
    return model


def fix_pos(x_cord):
    x = 87 - ord(x_cord.upper())
    return x


def train(path, save_path="data/rssi.hd5", epochs=1000, batches=1000, verbose=0):
    x = read_csv(path, index_col=None)

    x['x'] = x['location'].str[0]
    x['y'] = x['location'].str[1:]
    x.drop(["location"], axis=1, inplace=True)
    x["x"] = x["x"].apply(fix_pos)
    x["y"] = x["y"].astype(int)

    y = x.iloc[:, -2:]
    x = x.iloc[:, 1:-2]
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=.2, shuffle=False)
    es = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto', restore_best_weights=True)
    model = create_deep(train_x.shape[1])
    model.fit(x=train_x,
              y=train_y,
              validation_data=(val_x, val_y),
              epochs=epochs,
              batch_size=batches,
              verbose=verbose,
              callbacks=[es])

    preds = model.predict(val_x)
    l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))
    print("Error: ", l2dists_mean)
    model.save(save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='server for handling websockets')
    parser.add_argument('train_path', type=str, help="host name")
    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--batch-size', type=int, default=1000, help="batch size for training")
    parser.add_argument('--verbose', type=int, default=0, help="Set verbose level")
    parser.add_argument('--save-path', type=str, default="data/rssi.hd5",
                        help="Set verbose level")
    args = parser.parse_args()
    train_path = args.train_path
    train(train_path, args.save_path, epochs=args.epochs, batches=args.batch_size, verbose=args.verbose)
