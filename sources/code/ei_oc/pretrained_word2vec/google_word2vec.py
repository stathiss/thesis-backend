# General inputs
import numpy as np
import logging
import sys
import multiprocessing

# NumPy Imports
from numpy import array
from numpy import zeros

# Gensim and nltk inputs
from gensim.models.keyedvectors import KeyedVectors
from gensim.corpora.dictionary import Dictionary

# Shuffle
from random import shuffle

# TensorFlow inputs
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# My code inputs
from sources.loaders.loaders import parse_dataset
from sources.preprocessing.preprocessing import tweet_tokenizer
from sources.utils import get_pearson_correlation, write_predictions


np.random.seed(1500)  # For Reproducibility
dict_and_embs = 'dict_end_embs'
log = logging.getLogger()

# log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

vocab_dim = 118717
num_words = 20000
pad_words = 65
maxlen = 300
window_size = 12
batch_size = 32
n_epoch = 50
input_length = 100
cpu_count = multiprocessing.cpu_count()

log.info('source load')


def tokenizer(text):
    text = [clean_tweet(document) for document in text]
    return text


def clean_tweet(tweet):
    mod_tweet = tweet.split()
    return mod_tweet


def sentences_perm(sentences):
    shuffle(sentences)
    return sentences


def import_tag(datasets=None):
    if datasets is not None:
        train = {}
        test = {}
        for k, data in datasets.items():
            for val, each_line in enumerate(data):
                if k == 'train':
                    train[val] = each_line
                else:
                    test[val] = each_line
        return train, test
    else:
        print('Data not found...')


def create_dictionaries(train=None, test=None, model=None):
    if (train is not None) and (model is not None) and (test is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_the_dataset(data):
            for key in data.keys():
                txt = data[key].lower().replace('\n', '').split()
                new_txt = []
                for word in txt:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data[key] = new_txt
            return data

        train = parse_the_dataset(train)
        test = parse_the_dataset(test)
        return w2indx, w2vec, train, test
    else:
        print('No data provided...')


def google_word2vec_model(emotion):

    print('Load data...')
    X_train = tweet_tokenizer('EI-reg', emotion, 'train')
    y_train = array(parse_dataset('EI-reg', emotion, 'train')[3])
    X_test = tweet_tokenizer('EI-reg', emotion, 'development')
    y_test = array(parse_dataset('EI-reg', emotion, 'development')[3])
    dev_dataset = parse_dataset('EI-reg', emotion, 'development')

    print('Tokenising...')
    t = Tokenizer(num_words=num_words, lower=True)
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

    print('Load pretrained GooGle Word2vec model...')
    word_vectors = KeyedVectors.load_word2vec_format(
        'sources/features/pretrained_vectors/GoogleNews-vectors-negative300.bin', binary=True)

    vocabulary_size = min(len(t.word_index) + 1, num_words)
    embedding_matrix = np.zeros((vocabulary_size, maxlen))
    for word, i in t.word_index.items():
        if i >= num_words:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), maxlen)

    del (word_vectors)

    print('Defining a Simple Keras Model...')
    lstm_model = Sequential()  # or Graph
    lstm_model.add(Embedding(output_dim=maxlen,
                             input_dim=vocab_size,
                             mask_zero=True,
                             embeddings_initializer=Constant(embedding_matrix),
                             input_length=input_length))  # Adding Input Length
    lstm_model.add(LSTM(maxlen))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(128, kernel_initializer='normal', activation='sigmoid'))

    # The Hidden Layers :
    lstm_model.add(Dense(256, kernel_initializer='normal', activation='sigmoid'))
    lstm_model.add(Dense(256, kernel_initializer='normal', activation='sigmoid'))
    lstm_model.add(Dense(256, kernel_initializer='normal', activation='sigmoid'))

    # The Output Layer :
    lstm_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile the network :
    print('Compiling the Model...')

    lstm_model.compile(loss='mse',
                       optimizer='adam',
                       metrics=['mae'])

    print('Summary...')
    lstm_model.summary()

    print("Train...")
    lstm_model.fit(X_train, y_train,
                   batch_size=32,
                   epochs=n_epoch,
                   validation_split=0.2,
                   validation_data=(X_test, y_test))

    print("Evaluate...")
    score = lstm_model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)

    predictions = lstm_model.predict(X_test)
    predictions = [prediction[0] for prediction in predictions]
    file_name = './dumps/EI-reg_en_' + emotion + '_deep_learning_.txt'
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation('1',
                                  file_name,
                                  'datasets/EI-reg/development_set/2018-EI-reg-En-' + emotion + '-dev.txt'))
