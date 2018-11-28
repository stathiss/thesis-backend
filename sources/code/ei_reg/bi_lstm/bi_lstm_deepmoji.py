from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Embedding, GRU
from keras.optimizers import Adam
from numpy import array
from keras.models import load_model
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.loaders.loaders import parse_dataset
from sources.utils import get_pearson_correlation, write_predictions
import pickle


# return training data
def get_train():
    seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
    seq = array(seq)
    X, y = seq[:, 0], seq[:, 1]
    X = X.reshape((len(X), 1, 1))
    return X, y


def predict():
    # fit model
    with open('dumps/deepmoji_vector_joy_train', 'rb') as fp:
        X = pickle.load(fp)
    y = array(parse_dataset('EI-reg', 'joy', 'train')[3])
    X = X.reshape((len(X), 2304, 1))
    print(X)

    with open('dumps/deepmoji_vector_joy_dev', 'rb') as fp:
        X_test = pickle.load(fp)
        X = X.reshape((len(X), 2304, 1))
    y_test = parse_dataset('EI-reg', 'joy', 'development')[3]
    # define model
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(2304, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=3, batch_size=1, verbose=2)

    # make predictions
    prediction = model.evaluate(X_test, y_test)
    print(prediction)
    file_name = './dumps/EI-reg_en_joy_pred_dev_bilstm.txt'
    write_predictions(file_name, y_test, prediction)
    print(file_name)
    print(get_pearson_correlation('1', file_name, 'datasets/EI-reg/development_set/2018-EI-reg-En-joy-dev.txt'))
    print(get_pearson_correlation('1', file_name, 'datasets/EI-reg/development_set/2018-EI-reg-En-joy-dev.txt')[0])
