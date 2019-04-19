# n_observations = 19938
# n_features = 520
# distances = np.zeroes((1,n_observations))

# for i in range(1,n_observations):

from sklearn.neighbors import KNeighborsRegressor
from train_lat_long_detector import load
from sklearn import metrics

train_csv_path = './data/uji_data/1478167720_9233432_trainingData.csv'
valid_csv_path = './data/uji_data/1478167721_0345678_validationData.csv'
# Load data
train_x, train_y, valid_x, valid_y, test_x, test_y = load(train_csv_path, valid_csv_path)
neigh = KNeighborsRegressor(n_neighbors=3)

print(train_y[:, :2].shape)
neigh.fit(train_x, train_y[:, :2])

pr = neigh.predict(valid_x)
pr2 = neigh.predict(test_x)
print(neigh.predict(valid_x[0:10,:]))
print("Actual")
print(train_y[:, :2])
print("\n Accuracy")
print(metrics.mean_squared_error(valid_y[:,:2], pr))
print(metrics.mean_squared_error(test_y[:,:2], pr2))


