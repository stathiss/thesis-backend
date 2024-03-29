import multiprocessing
import logging
import sys
import numpy
from numpy import array, asarray, zeros
from scipy import stats

# TensorFlow and Keras inputs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional, Dense, Dropout, LSTM
from keras.initializers import glorot_normal

# My code inputs
from sources.loaders.loaders import parse_dataset
from sources.loaders.files import find_path
from sources.preprocessing.preprocessing import tweet_tokenizer
from sources.utils import predictions_of_file, pearson_correlation_loss, write_predictions, Attention


numpy.random.seed(1500)  # For Reproducibility
log = logging.getLogger()

# log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

pad_words = 65
n_epoch = 10
input_length = 100
window_size = 12
batch_size = 64
validation_split = 0.1
cpu_count = multiprocessing.cpu_count()

log.info('source load')


def glove_attention_model(emotion, maxlen):
    train_file = 'train'
    test_file = 'development'

    print('Load data...')
    x_train = tweet_tokenizer('EI-reg', emotion, train_file)
    y_train = array(parse_dataset('EI-reg', emotion, train_file)[3])
    x_test = tweet_tokenizer('EI-reg', emotion, test_file)
    y_test = array(parse_dataset('EI-reg', emotion, test_file)[3])

    dev_dataset = parse_dataset('EI-reg', emotion, test_file)

    print('Tokenising...')
    t = Tokenizer()
    t.fit_on_texts(x_train + x_test)
    vocab_size = len(t.word_counts) + 1

    print('Integer encoding...')
    encoded_train = t.texts_to_sequences(x_train)
    encoded_dev = t.texts_to_sequences(x_test)

    print('Padding documents in length of {} words...'.format(pad_words))
    padded_train = pad_sequences(encoded_train, maxlen=pad_words)
    padded_dev = pad_sequences(encoded_dev, maxlen=pad_words)

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
                              input_length=pad_words,
                              ))  # Adding Input Length
    model_glove.layers[0].set_weights([embedding_matrix])
    model_glove.add(Bidirectional(LSTM(300, return_sequences=True, dropout=0.25, recurrent_dropout=0.25)))
    model_glove.add(Attention(65))
    model_glove.add(Dropout(0.5))
    model_glove.add(Dense(128, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    model_glove.add(Dense(128, activation='relu', kernel_initializer=glorot_normal(seed=None)))

    # The Hidden Layers :
    model_glove.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    model_glove.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    model_glove.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))
    model_glove.add(Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=None)))

    # The Output Layer :
    model_glove.add(Dense(1, activation='relu', kernel_initializer=glorot_normal(seed=None)))

    # Compile the network :
    print('Compiling the Model...')
    model_glove.compile(loss='mean_squared_error', optimizer='adam', metrics=[pearson_correlation_loss])

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
    print('Test score:', (score[0] - 1) * (-1), score[1])
    # Calculate Predictions
    predictions = model_glove.predict(padded_dev)

    print('Pearson Correlation...')
    predictions = [prediction[0] for prediction in predictions]

    # Write Predictions
    file_name = "./dumps/EI-reg/" + test_file + "/BiLSTM+Att/EI-reg_en_" + emotion + "_glove_" + str(maxlen) + ".txt"
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)

    real_golden = predictions_of_file(find_path('EI-reg', emotion, test_file))
    print(stats.pearsonr(predictions, real_golden)[0])
