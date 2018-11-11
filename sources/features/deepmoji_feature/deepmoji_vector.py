# -*- coding: utf-8 -*-

"""
Use DeepMoji to encode texts into emotional feature vectors.
"""

from __future__ import division
import json
import pickle
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_feature_encoding
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from sources.preprocessing.preprocessing import tweet_tokenizer


def deepmoji_vector(task, emotion, label):
    np.set_printoptions(threshold=np.nan)

    TEST_SENTENCES = ['@deodevine6 i can\'t bully you and niall impossible', '@BrumSciComm in a danish.  What horror, what depths. #gbbo']
    maxlen = 30
    batch_size = 64

    # print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    # print('st', st)
    tokenized, a, b = st.tokenize_sentences(TEST_SENTENCES)

    # print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = deepmoji_feature_encoding(maxlen, PRETRAINED_PATH)
    model.summary()

    # print('Encoding texts..')

    encoding = model.predict(tokenized)

    # print('First 5 dimensions for sentence: {}'.format(TEST_SENTENCES[0]))
    # print(encoding[0])
    # print(len(encoding[0]))
    with open('dumps/encoding', 'wb') as fp:
        pickle.dump(encoding, fp)
    # Now you could visualize the encodings to see differences,
    # run a logistic regression classifier on top,
    # or basically anything you'd like to do.
    return encoding


def read_test():
    with open('dumps/encoding', 'rb') as fp:
        itemlist = pickle.load(fp)
        print(itemlist)
