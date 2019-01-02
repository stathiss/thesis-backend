from sources.code.ei_reg.my_word2vec_lstm.my_word2vec_lstm import my_word2vec_model
from sources.code.ei_reg.glove_vectors.glove_vectors import glove_model
from sources.code.ei_reg.pretrained_word2vec.google_word2vec import google_word2vec_model
from sources.code.ei_reg.random_forest.random_forest_deepmoji import predict_random_forest__deepmoji
from sources.utils import ensemble_predictions

# predict_random_forest__deepmoji('fear')
# my_word2vec_model('fear')
# glove_model('fear')
# google_word2vec_model('fear')

ensemble_predictions(
    ['dumps/EI-reg_en_fear_test_glove_vectors_100.txt',
     'dumps/EI-reg_en_fear_test_google_vectors.txt',
     'dumps/EI-reg_en_fear_test_svr.txt',
     'dumps/EI-reg_en_fear_test_random_forest.txt'],
    [0.25, 0.25, 0.25, 0.25],
    'EI-reg',
    'fear',
    'gold-no-mystery',
    'datasets/gold-labels/EI-reg/2018-EI-reg-En-fear-test-gold-no-mystery.txt')
