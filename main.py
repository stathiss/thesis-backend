from sources.code.ei_reg.my_word2vec_lstm.my_word2vec_lstm import my_word2vec_model
from sources.code.ei_reg.glove_vectors.glove_vectors import glove_model
from sources.code.ei_reg.google_word2vec.google_word2vec import google_word2vec_model
from sources.code.ei_reg.random_forest.random_forest_deepmoji import predict_random_forest__deepmoji
from sources.code.ei_reg.attention_layer.google_word2vec_attention import google_word2vec_attention_model
from sources.utils import ensemble_predictions

# predict_random_forest__deepmoji('fear')
# my_word2vec_model('fear')
# glove_model('fear')
# google_word2vec_model('fear')
# google_word2vec_attention_model('fear')

ensemble_predictions(
    ['dumps/EI-reg_en_fear_glove_vectors.txt',
     'dumps/EI-reg_en_fear_google_vectors.txt',
     'dumps/EI-reg_en_fearsvr.txt',
     'dumps/EI-reg_en_sadness_pred_dev_random_forest.txt',
     'dumps/EI-reg_en_fear_dev_google_attention_vectors.txt'],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    'EI-reg',
    'fear',
    'development',
    'datasets/EI-reg/development_set/2018-EI-reg-En-fear-dev.txt')
