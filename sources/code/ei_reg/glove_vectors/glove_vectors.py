import pickle
import multiprocessing
import logging
import sys
import tensorflow as tf
import numpy
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# TensorFlow inputs
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.layers import Bidirectional, Dense, Dropout, LSTM

# My code inputs
from sources.loaders.loaders import parse_dataset
from sources.preprocessing.preprocessing import tweet_tokenizer
from sources.utils import get_pearson_correlation, write_predictions

from keras import backend as K

numpy.random.seed(1500)  # For Reproducibility
log = logging.getLogger()

# log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

maxlen = 300
pad_words = 65
n_epoch = 25
input_length = 100
window_size = 12
batch_size = 64
validation_split = 0.1
cpu_count = multiprocessing.cpu_count()

log.info('source load')


def pearson_correlation_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return -r


def glove_model(emotion):

    print('Load data...')
    X_train = tweet_tokenizer('EI-reg', emotion, 'train')
    y_train = array(parse_dataset('EI-reg', emotion, 'train')[3])
    X_test = tweet_tokenizer('EI-reg', emotion, 'development')
    y_test = array(parse_dataset('EI-reg', emotion, 'development')[3])
    dev_dataset = parse_dataset('EI-reg', emotion, 'development')

    print('Tokenising...')
    t = Tokenizer()
    t.fit_on_texts(X_train + X_test)
    vocab_size = len(t.word_counts) + 1

    print('Integer encoding...')
    encoded_train = t.texts_to_sequences(X_train)
    encoded_dev = t.texts_to_sequences(X_test)
    print(encoded_dev)
    print(encoded_train)

    print('Padding documents in length of {} words...'.format(pad_words))
    padded_train = pad_sequences(encoded_train, maxlen=pad_words)
    padded_dev = pad_sequences(encoded_dev, maxlen=pad_words)
    print(padded_train)

    print('Load pretrained data')
    embeddings_index = dict()
    f = open('sources/features/pretrained_vectors/glove.6B.' + str(maxlen) + 'd.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))

    print('Creating weight matrix...')
    embedding_matrix = zeros((vocab_size, maxlen))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print('Defining the model...')
    model_glove = Sequential()  # or Graph
    model_glove.add(Embedding(output_dim=maxlen,
                              input_dim=vocab_size,
                              mask_zero=True,
                              input_length=pad_words
                              ))  # Adding Input Length
    model_glove.layers[0].set_weights([embedding_matrix])
    model_glove.add(Bidirectional(LSTM(300)))
    model_glove.add(Dropout(0.5))
    model_glove.add(Dense(128, activation='relu'))
    model_glove.add(Dense(128, activation='relu'))

    # The Hidden Layers :
    model_glove.add(Dense(256, activation='relu'))
    model_glove.add(Dense(256, activation='relu'))
    model_glove.add(Dense(256, activation='relu'))
    model_glove.add(Dense(256, activation='relu'))

    # The Output Layer :
    model_glove.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile the network :
    print('Compiling the Model...')
    model_glove.compile(loss=pearson_correlation_loss,
                        optimizer='adam',
                        metrics=['mae'])

    print('Summary...')
    model_glove.summary()

    print("Train...")
    model_glove.fit(padded_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epoch,
                    validation_split=validation_split,
                    validation_data=(padded_dev, y_test))

    print("Evaluate...")
    score = model_glove.evaluate(padded_dev, y_test, batch_size=batch_size)
    print('Test score:', score)

    predictions = model_glove.predict(padded_dev)

    print('Pearson Correlation...')
    print(predictions)
    predictions = [prediction[0] for prediction in predictions]
    print(predictions)
    file_name = './dumps/EI-reg_en_' + emotion + '_deep_learning_.txt'
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation('1',
                                  file_name,
                                  'datasets/EI-reg/development_set/2018-EI-reg-En-' + emotion + '-dev.txt'))
