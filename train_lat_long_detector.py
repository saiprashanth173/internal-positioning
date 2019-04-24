import bisect
import os
from abc import ABCMeta
from math import sqrt

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

CV = 2
FEATURES = 520
train_csv_path = './data/uji_data/1478167720_9233432_trainingData.csv'
valid_csv_path = './data/uji_data/1478167721_0345678_validationData.csv'
longitude_scaler_encoder = './data/uji_models/longitude.pkl'
latitude_scaler_encoder = './data/uji_models/latitude.pkl'
pca_encoder = './data/uji_models/pca.pkl'

# Normalize scaler
if os.path.exists(longitude_scaler_encoder) and os.path.exists(latitude_scaler_encoder):
    longitude_scaler = joblib.load(longitude_scaler_encoder)
    latitude_scaler = joblib.load(latitude_scaler_encoder)
else:
    longitude_scaler = MinMaxScaler()
    latitude_scaler = MinMaxScaler()

if os.path.exists(pca_encoder):
    pca = joblib.load(pca_encoder)
else:
    pca = PCA()


def get_x_y_from_df(df):
    X = df.get_values().T[:520].T
    y = df.get_values().T[[520, 521, 522, 523], :].T
    return X, y


def load(train_file_name, valid_file_name):
    # Read the file
    if train_file_name == None or valid_file_name == None:
        print('file name is None...')
        exit()
    train_data_frame = pd.read_csv(train_file_name)
    test_data_frame = pd.read_csv(valid_file_name)

    # Random pick 1/10 data to be the final validation data
    rest_data_frame = train_data_frame
    valid_data_trame = pd.DataFrame(columns=train_data_frame.columns)
    valid_num = int(len(train_data_frame) / 10)
    sample_row = rest_data_frame.sample(valid_num)
    rest_data_frame = rest_data_frame.drop(sample_row.index)
    valid_data_trame = valid_data_trame.append(sample_row)
    train_data_frame = rest_data_frame

    # Split data frame and return
    training_x = train_data_frame.get_values().T[:520].T
    training_y = train_data_frame.get_values().T[[520, 521, 522, 523], :].T
    validation_x = valid_data_trame.get_values().T[:520].T
    validation_y = valid_data_trame.get_values().T[[520, 521, 522, 523], :].T
    testing_x = test_data_frame.get_values().T[:520].T
    testing_y = test_data_frame.get_values().T[[520, 521, 522, 523], :].T
    return training_x, training_y, validation_x, validation_y, testing_x, testing_y


def normalizeX(arr, expl_var=0.80, do_PCA=False):
    global FEATURES
    res = np.copy(arr).astype(np.float)
    if do_PCA == False:
        res[res == 100] = 0
        res = -0.01 * res
        return res
    else:
        print("PCA")
        try:
            transformed = pca.transform(arr)
            var = pca.explained_variance_ratio_.cumsum()
            FEATURES = bisect.bisect(var, expl_var)
            print(arr.shape, transformed.shape)
            return transformed[:, :FEATURES + 1]

        except NotFittedError as e:
            pca.fit(res)
            joblib.dump(pca, pca_encoder)
            var = pca.explained_variance_ratio_.cumsum()
            FEATURES = bisect.bisect(var, expl_var)

        return pca.transform(res)[:, : FEATURES + 1]


def normalizeY(longitude_arr, latitude_arr):
    global longitude_scaler
    global latitude_scaler
    longitude_arr = np.reshape(longitude_arr, [-1, 1])
    latitude_arr = np.reshape(latitude_arr, [-1, 1])
    longitude_scaler.fit(longitude_arr)
    latitude_scaler.fit(latitude_arr)
    joblib.dump(longitude_scaler, longitude_scaler_encoder)
    joblib.dump(latitude_scaler, latitude_scaler_encoder)
    return np.reshape(longitude_scaler.transform(longitude_arr), [-1]), \
           np.reshape(latitude_scaler.transform(latitude_arr), [-1])


def reverse_normalizeY(longitude_arr, latitude_arr):
    global longitude_scaler
    global latitude_scaler
    longitude_arr = np.reshape(longitude_arr, [-1, 1])
    latitude_arr = np.reshape(latitude_arr, [-1, 1])
    return np.reshape(longitude_scaler.inverse_transform(longitude_arr), [-1]), \
           np.reshape(latitude_scaler.inverse_transform(latitude_arr), [-1])


def getMiniBatch(arr, batch_size=3):
    index = 0
    while True:
        # print index + batch_size
        if index + batch_size >= len(arr):
            res = arr[index:]
            res = np.concatenate((res, arr[:index + batch_size - len(arr)]))
        else:
            res = arr[index:index + batch_size]
        index = (index + batch_size) % len(arr)
        yield res


class AbstractModel(object):
    __metaclass__ = ABCMeta

    # Model save path
    parameter_save_path = 'param.pkl'
    longitude_regression_model_save_path = None
    latitude_regression_model_save_path = None

    # ML model object
    longitude_regression_model = None
    latitude_regression_model = None

    # Normalize variable
    longitude_mean = None
    longitude_std = None
    latitude_mean = None
    latitude_std = None
    longitude_shift_distance = None
    latitude_shift_distance = None

    # Training data
    normalize_x = None
    longitude_normalize_y = None
    latitude_normalize_y = None

    def __init__(self, do_PCA=False):
        self.do_PCA = do_PCA
        pass

    def _preprocess(self, x, y):
        self.normalize_x = normalizeX(x, do_PCA=self.do_PCA)
        self.longitude_normalize_y, self.latitude_normalize_y = normalizeY(y[:, 0], y[:, 1])

    def save(self):
        print
        "<< Saving >>"
        joblib.dump(self.longitude_regression_model, self.longitude_regression_model_save_path)
        joblib.dump(self.latitude_regression_model, self.latitude_regression_model_save_path)

    def load(self):
        self.longitude_regression_model = joblib.load(self.longitude_regression_model_save_path)
        self.latitude_regression_model = joblib.load(self.latitude_regression_model_save_path)

    def fit(self, x, y, do_PCA=False):
        # Data pre-processing
        self._preprocess(x, y)

        # Train the model
        print("<< training >>")
        print(self.normalize_x.shape)
        self.longitude_regression_model.fit(self.normalize_x, self.longitude_normalize_y)
        self.latitude_regression_model.fit(self.normalize_x, self.latitude_normalize_y)

        # Release the memory
        del self.normalize_x
        del self.longitude_normalize_y
        del self.latitude_normalize_y

        # Save the result
        self.save()

    def predict(self, x):
        # Load model
        self.load()

        # Testing
        x = normalizeX(x, do_PCA=self.do_PCA)
        predict_longitude = self.longitude_regression_model.predict(x)
        predict_latitude = self.latitude_regression_model.predict(x)

        # Reverse normalization
        predict_longitude, predict_latitude = reverse_normalizeY(predict_longitude, predict_latitude)

        # Return the result
        res = np.concatenate((np.expand_dims(predict_longitude, axis=-1),
                              np.expand_dims(predict_latitude, axis=-1)), axis=-1)

        return res

    def error(self, x, y):
        _y = self.predict(x)
        print("Longitude: ", mean_squared_error(_y[:, 0], y[:, 0]))
        print("Latitude: ", mean_squared_error(_y[:, 1], y[:, 1]))
        print("Root Mean Square Error", sqrt(mean_squared_error(_y[:, :2], y[:, :2])))
        print("Mean Square Error", mean_squared_error(_y[:, :2], y[:, :2]))
        print("Mean Absolute Error", mean_absolute_error(_y[:, :2], y[:, :2]))
        print("R2", r2_score(_y[:, :2], y[:, :2]))
        print("Explained Variance", explained_variance_score(_y[:, :2], y[:, :2]))


class SVM(AbstractModel):
    # Model save path
    longitude_regression_model_save_path = './data/uji_models/svm_long.pkl'
    latitude_regression_model_save_path = './data/uji_models/svm_lat.pkl'

    def __init__(self, do_PCA=False):
        Cs = [0.001, 0.01]
        gammas = [0.001, 0.01]
        param_grid = {'C': Cs, 'gamma': gammas, 'kernel': ['linear', 'rbf']}
        longitude_reg_model = GridSearchCV(SVR(), param_grid, cv=CV, verbose=True)
        lat_reg_model = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=CV)
        self.longitude_regression_model = longitude_reg_model
        self.latitude_regression_model = lat_reg_model
        super(SVM, self).__init__(do_PCA=do_PCA)


class RandomForest(AbstractModel):
    # Model save path
    longitude_regression_model_save_path = './data/uji_models/rf_long.pkl'
    latitude_regression_model_save_path = './data/uji_models/rf_lat.pkl'

    def __init__(self, do_PCA=False):
        self.longitude_regression_model = RandomForestRegressor()
        self.latitude_regression_model = RandomForestRegressor()
        super(RandomForest, self).__init__(do_PCA=do_PCA)


class GradientBoostingDecisionTree(AbstractModel):
    # Model save path
    longitude_regression_model_save_path = './data/uji_models/gb_long.pkl'
    latitude_regression_model_save_path = './data/uji_models/gb_lat.pkl'

    def __init__(self, do_PCA=False):
        self.longitude_regression_model = GradientBoostingRegressor()
        self.latitude_regression_model = GradientBoostingRegressor()
        super(GradientBoostingDecisionTree, self).__init__(do_PCA=do_PCA)


class KNN(AbstractModel):
    # Model save path
    longitude_regression_model_save_path = './data/uji_models/knnb_long.pkl'
    latitude_regression_model_save_path = './data/uji_models/knnb_lat.pkl'

    def __init__(self, do_PCA=False):
        self.longitude_regression_model = KNeighborsRegressor()
        self.latitude_regression_model = KNeighborsRegressor()
        super(KNN, self).__init__(do_PCA=do_PCA)


class XGradientBoostingDecisionTree(AbstractModel):
    # Model save path
    longitude_regression_model_save_path = './data/uji_models/xgb_long.pkl'
    latitude_regression_model_save_path = './data/uji_models/xgb_lat.pkl'

    def __init__(self, do_PCA=False):
        self.longitude_regression_model = XGBRegressor()
        self.latitude_regression_model = XGBRegressor()
        super(XGradientBoostingDecisionTree, self).__init__(do_PCA=do_PCA)


class ComplexDNN(AbstractModel):
    # Model save path
    longitude_regression_model_save_path = './data/uji_models/complexDnn_long.pkl'
    latitude_regression_model_save_path = './data/uji_models/complexDnn_lat.pkl'

    sess = None

    def __init__(self, do_PCA=False):
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, [None, FEATURES])
        self.locating_y = tf.placeholder(tf.float32, [None, 2])
        self.alternative_ctl = tf.placeholder(tf.bool)

        locating_network = tl.layers.InputLayer(self.x, name='Input')
        locating_network = tl.layers.DenseLayer(locating_network, n_units=2048, act=tf.nn.relu, name='locating_fc1')
        locating_network = tl.layers.DenseLayer(locating_network, n_units=256, act=tf.nn.relu, name='locating_fc2')
        locating_network = tl.layers.DenseLayer(locating_network, n_units=128, act=tf.nn.relu, name='locating_fc3')
        locating_network = tl.layers.DenseLayer(locating_network, n_units=2, act=tf.identity, name='locating_fc4')
        self.locating_predict_y = locating_network.outputs
        self.locating_cost = tl.cost.mean_squared_error(self.locating_y, self.locating_predict_y)
        self.locating_optimize = tf.train.AdamOptimizer().minimize(self.locating_cost)
        super(ComplexDNN, self).__init__(do_PCA=do_PCA)

    def fit(self, x, y, epoch=10, batch_size=256):
        # Data pre-processing
        self._preprocess(x, y)

        location_pair = np.concatenate((
            np.expand_dims(self.longitude_normalize_y, -1), np.expand_dims(self.latitude_normalize_y, -1)
        ), axis=-1)

        # Train the model
        print
        "<< training >>"
        self.sess.run(tf.global_variables_initializer())

        for k in range(10):
            print("-------- epoch ", k, ' ---------')
            print("\n< position >\n")
            for i in range(epoch):
                mini_x = getMiniBatch(self.normalize_x, batch_size)
                mini_y = getMiniBatch(location_pair, batch_size)
                feed_dict = {
                    self.x: next(mini_x),
                    self.locating_y: next(mini_y)
                }
                _cost, _, _output = self.sess.run([self.locating_cost, self.locating_optimize, self.locating_predict_y],
                                                  feed_dict=feed_dict)
                if i % 100 == 0:
                    print("epoch: ", i, '\tcost: ', _cost)

        self.save()

    def save(self, save_path='./complex_dnn.ckpt'):
        super(ComplexDNN, self).save()
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def predict(self, x, model_path='./complex_dnn.ckpt'):
        # Load model and parameter
        super(ComplexDNN, self).load()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

        # Testing
        x = normalizeX(x)
        predict_result = self.sess.run(self.locating_predict_y, feed_dict={self.x: x})
        predict_longitude = predict_result[:, 0]
        predict_latitude = predict_result[:, 1]

        # Reverse normalization
        predict_longitude, predict_latitude = reverse_normalizeY(predict_longitude, predict_latitude)

        # Return the result
        res = np.concatenate((np.expand_dims(predict_longitude, axis=-1),
                              np.expand_dims(predict_latitude, axis=-1)), axis=-1)
        return res


if __name__ == '__main__':
    # Parsing arguments
    import argparse

    parser = argparse.ArgumentParser(description='Models for finding latitude and longitude')
    argument = parser.add_argument('model_name', type=str, help="model name: SVM, RF, GBDT, CDNN")
    parser.add_argument('-y', action='store_true', help="PCA Mode")
    args = parser.parse_args()
    model_name = args.model_name
    # Load data
    train_x, train_y, valid_x, valid_y, test_x, test_y = load(train_csv_path, valid_csv_path)
    pca_mode = args.y

    # # # Training
    if model_name == 'SVM':
        svm_model = SVM(do_PCA=pca_mode)
        svm_model.fit(train_x, train_y)
        model = svm_model
        print('SVM error: ', svm_model.error(test_x, test_y))
    elif model_name == 'RF':
        rf_model = RandomForest(do_PCA=pca_mode)
        rf_model.fit(train_x, train_y)
        model = rf_model
        print('RF error: ', rf_model.error(test_x, test_y))
    elif model_name == 'GBDT':
        gbdt_model = XGradientBoostingDecisionTree(do_PCA=pca_mode)
        gbdt_model.fit(train_x, train_y)
        print('Gradient boosting decision tree error: ', gbdt_model.error(test_x, test_y))
        model = gbdt_model
    elif model_name == 'CDNN':
        dnn_model = ComplexDNN(do_PCA=pca_mode)
        dnn_model.fit(train_x, train_y)
        model = dnn_model
        print('DNN error: ', dnn_model.error(test_x, test_y))
    else:
        print('Model Name is incorrect')

    print("Validation Accuracy!!!!")
    model.error(valid_x, valid_y)

    print("Test Accuracy!!!!")
    model.error(test_x, test_y)
