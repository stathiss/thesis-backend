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

import scipy.stats

numpy.random.seed(1500)  # For Reproducibility
dict_and_embs = 'dict_end_embs'
log = logging.getLogger()

# log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

maxlen = 100
pad_words = 100
n_epoch = 50
input_length = 100
window_size = 12
batch_size = 32
cpu_count = multiprocessing.cpu_count()

log.info('source load')


def glove_model(emotion):
    print('Load data...')
    X_train = tweet_tokenizer('EI-reg', emotion, 'train')
    print(X_train)
    with open('dumps/tweet_tokenizer_' + emotion + '_train', 'wb') as fp:
        pickle.dump(X_train, fp)
    y_train = array(parse_dataset('EI-reg', emotion, 'train')[3])
    print(y_train[0])
    X_test = tweet_tokenizer('EI-reg', emotion, 'development')
    with open('dumps/tweet_tokenizer_' + emotion + '_dev', 'wb') as fp:
        pickle.dump(X_test, fp)
    y_test = array(parse_dataset('EI-reg', emotion, 'development')[3])
    dev_dataset = parse_dataset('EI-reg', emotion, 'development')

    print('Tokenising...')
    t = Tokenizer()
    t.fit_on_texts(X_train + X_test)
    vocab_size = len(t.word_index) + 1

    print('Integer encoding...')
    encoded_train = t.texts_to_sequences(X_train)
    encoded_dev = t.texts_to_sequences(X_test)
    print(encoded_dev)
    print(encoded_train)

    print('Padding documents in length of {} words...'.format(pad_words))
    max_length = pad_words
    padded_train = pad_sequences(encoded_train, maxlen=max_length)
    padded_dev = pad_sequences(encoded_dev, maxlen=max_length)
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
    print('length', len(embedding_matrix))
    print(embedding_matrix)
    model_glove = Sequential()  # or Graph
    model_glove.add(Embedding(output_dim=maxlen,
                              input_dim=vocab_size,
                              mask_zero=True,
                              embeddings_initializer=Constant(embedding_matrix),
                              input_length=pad_words
                              ))  # Adding Input Length
    model_glove.add(Bidirectional(LSTM(196)))
    model_glove.add(Dropout(0.1))

    # The Output Layer :
    model_glove.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    print('Compiling the Model...')
    model_glove.compile(loss=['mse'],
                        optimizer='adam',
                        metrics=['mae'])

    print('Summary...')
    model_glove.summary()
    for i in padded_dev:
        if None in i:
            print('erorrrroro')

    print("Train...")
    model_glove.fit(padded_train, y_train,
                    batch_size=32,
                    epochs=n_epoch,
                    validation_split=0.2,
                    validation_data=(padded_dev, y_test))

    print("Evaluate...")
    score = model_glove.evaluate(padded_dev, y_test, batch_size=32)
    print('Test score:', score)

    predictions = model_glove.predict(padded_dev)
    print(predictions)
    predictions = [prediction[0] for prediction in predictions]
    print(predictions)
    file_name = './dumps/EI-reg_en_' + emotion + '_deep_learning_.txt'
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation('1',
                                  file_name,
                                  'datasets/EI-reg/development_set/2018-EI-reg-En-' + emotion + '-dev.txt'))
