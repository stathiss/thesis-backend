# General inputs
import numpy as np
import logging
import sys
import multiprocessing
import pickle

# Gensim and nltk inputs
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary

# Shuffle
from random import shuffle

# TensorFlow inputs
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.initializers import glorot_normal

# My code inputs
from sources.loaders.loaders import parse_dataset
from sources.loaders.files import find_path
from sources.preprocessing.preprocessing import tweet_tokenizer
from sources.utils import get_pearson_correlation, write_predictions, pearson_correlation_loss


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
maxlen = 100
window_size = 12
batch_size = 32
n_epoch = 15
input_length = 100
cpu_count = multiprocessing.cpu_count()
activation = 'relu'

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


def my_word2vec_model(emotion):
    train_file = 'train'
    test_file = 'development'

    X_train = tweet_tokenizer('EI-reg', emotion, train_file)
    y_train = parse_dataset('EI-reg', emotion, train_file)[3]

    X_test = tweet_tokenizer('EI-reg', emotion, test_file)
    y_test = parse_dataset('EI-reg', emotion, test_file)[3]

    dev_dataset = parse_dataset('EI-reg', emotion, test_file)
    print('Loading Data...')
    train, test = import_tag(datasets={'train': X_train, 'test': X_test})
    combined = train.values() + test.values()

    print('Tokenising...')
    combined = tokenizer(combined)

    # print combined

    print('Training a Word2vec model...')
    model = Word2Vec(size=maxlen,
                     window=window_size,
                     workers=cpu_count,
                     min_count=0)

    model.build_vocab(combined)

    for epoch in range(100):
        log.info('EPOCH: {}'.format(epoch))
        model.train(sentences_perm(combined), total_examples=model.corpus_count, epochs=model.epochs)

    print('Transform the Data...')
    index_dict, word_vectors, train, test = create_dictionaries(train=train,
                                                                test=test,
                                                                model=model)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index
    embedding_weights = np.zeros((n_symbols, maxlen))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
    print('Creating Datesets...')
    X_train = train.values()
    X_test = test.values()
    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Convert labels to Numpy Sets...')
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print('Defining a Simple Keras Model...')
    lstm_model = Sequential()  # or Graph
    lstm_model.add(Embedding(output_dim=maxlen,
                             input_dim=n_symbols,
                             mask_zero=True,
                             embeddings_initializer=Constant(embedding_weights),
                             input_length=input_length))  # Adding Input Length
    lstm_model.layers[0].set_weights([embedding_weights])
    lstm_model.add(Bidirectional(LSTM(300)))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(128, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    lstm_model.add(Dense(128, activation='relu', kernel_initializer=glorot_normal(seed=None)))

    # The Hidden Layers :
    lstm_model.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    lstm_model.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    lstm_model.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    lstm_model.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))

    # The Output Layer :
    lstm_model.add(Dense(1, kernel_initializer=glorot_normal(seed=None), activation='sigmoid'))

    # Compile the network :
    print('Compiling the Model...')
    lstm_model.compile(loss='mean_squared_error',
                       optimizer='adam',
                       metrics=['mae', pearson_correlation_loss])

    print('Summary...')
    lstm_model.summary()
    print("Train...")
    lstm_model.fit(X_train, y_train,
                   batch_size=32,
                   epochs=5,
                   validation_split=0.1,
                   validation_data=(X_test, y_test))

    print("Evaluate...")
    score = lstm_model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', (score[0] - 1) * (-1), score[1])

    predictions = lstm_model.predict(X_test)
    predictions = [prediction[0] for prediction in predictions]

    file_name = "./dumps/EI-reg/" + test_file + "/BiLSTM/EI-reg_en_" + emotion + "_my_model.txt"
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)

    del lstm_model
    print(get_pearson_correlation('1', file_name, find_path('EI-reg', emotion, test_file)))
