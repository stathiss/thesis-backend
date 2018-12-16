import numpy as np
import os
import pickle
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from sources.preprocessing.preprocessing import tweet_tokenizer_words


def get_init_embedding(dictionary):
    if not os.path.exists('./word_vectors.pickle'):
        glove_file = "sources/features/pretrained/glove.twitter.27B.50d.txt"
        word2vec_file = get_tmpfile("word2vec_format.vec")
        glove2word2vec(glove_file, word2vec_file)
        print("Loading Glove vectors...")
        word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)
        with open("./word_vectors.pickle", "wb") as f:
            pickle.dump(word_vectors, f)
    else:
        with open("./word_vectors.pickle", "rb") as f:
            word_vectors = pickle.load(f)

    reversed_dict = {value: key for key, value in dictionary.items()}

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros(50, dtype=np.float32)
        word_vec_list.append(word_vec)

    with open("embeddings.pickle", "wb") as f:
        pickle.dump(word_vec_list, f)
    return np.array(word_vec_list, dtype="float32")


def check_dict():
    counter = 0
    my_dict = {}
    tweets = tweet_tokenizer_words('EI-reg', 'sadness', 'train')
    print(len(tweets))
    for tweet in tweets:
        for word in tweet:
            if word in my_dict:
                continue
            else:
                my_dict[word] = counter
                counter += 1
    print(counter)
    print(my_dict)
