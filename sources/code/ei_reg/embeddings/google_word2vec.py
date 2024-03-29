# General inputs
import numpy as np
import logging
import sys
import multiprocessing
from scipy import stats

# NumPy Imports
from numpy import array
from numpy import zeros

# Gensim and nltk inputs
from gensim.models.keyedvectors import KeyedVectors

# Keras and TensorFlow inputs
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.initializers import glorot_normal
from keras.preprocessing.sequence import pad_sequences

# My code inputs
from sources.loaders.loaders import parse_dataset
from sources.loaders.files import find_path
from sources.preprocessing.preprocessing import tweet_tokenizer
from sources.utils import pearson_correlation_loss
from sources.utils import predictions_of_file, write_predictions

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
pad_words = 65
maxlen = 300
window_size = 12
batch_size = 32
n_epoch = 10
input_length = 100
cpu_count = multiprocessing.cpu_count()

log.info('source load')


def google_word2vec_model(emotion):
    train_file = 'train'
    test_file = 'development'

    print('Load data...')
    x_train = tweet_tokenizer('EI-reg', emotion, train_file)
    y_train = array(parse_dataset('EI-reg', emotion, train_file)[3])
    x_test = tweet_tokenizer('EI-reg', emotion, test_file)
    y_test = array(parse_dataset('EI-reg', emotion, test_file)[3])

    dev_dataset = parse_dataset('EI-reg', emotion, test_file)

    print('Tokenising...')
    t = Tokenizer(lower=True)
    t.fit_on_texts(x_train + x_test)
    vocab_size = len(t.word_counts) + 1
    num_words = vocab_size

    print('Integer encoding...')
    encoded_train = t.texts_to_sequences(x_train)
    encoded_dev = t.texts_to_sequences(x_test)

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
            embedding_matrix[i] = zeros(maxlen)

    del word_vectors

    print('Defining a Simple Keras Model...')
    google_model = Sequential()  # or Graph
    google_model.add(Embedding(output_dim=maxlen,
                               input_dim=vocab_size,
                               mask_zero=True,
                               embeddings_initializer=Constant(embedding_matrix),
                               input_length=pad_words))  # Adding Input Length
    google_model.layers[0].set_weights([embedding_matrix])
    google_model.add(Bidirectional(LSTM(300)))
    google_model.add(Dropout(0.5))
    google_model.add(Dense(128, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    google_model.add(Dense(128, activation='relu', kernel_initializer=glorot_normal(seed=None)))

    # The Hidden Layers :
    google_model.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    google_model.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    google_model.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    google_model.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))

    # The Output Layer :
    google_model.add(Dense(1, kernel_initializer=glorot_normal(seed=None), activation='sigmoid'))

    # Compile the network :
    print('Compiling the Model...')
    google_model.compile(loss='mean_squared_error',
                         optimizer='adam',
                         metrics=['mae', pearson_correlation_loss])

    print('Summary...')
    google_model.summary()
    print("Train...")
    google_model.fit(padded_train, y_train,
                     batch_size=32,
                     epochs=n_epoch,
                     validation_split=0.1,
                     validation_data=(padded_dev, y_test))

    print("Evaluate...")
    score = google_model.evaluate(padded_dev, y_test, batch_size=batch_size)
    print('Test score:', (score[0] - 1) * (-1), score[1])

    predictions = google_model.predict(padded_dev)
    predictions = [prediction[0] for prediction in predictions]
    real_golden = predictions_of_file(find_path('EI-reg', emotion, test_file))

    # Write Predictions
    file_name = "./dumps/EI-reg/" + test_file + "/BiLSTM/EI-reg_en_" + emotion + "_google.txt"
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)

    print(stats.pearsonr(predictions, real_golden)[0])
