from sources.code.ei_reg.embeddings.glove_vectors import glove_model
from sources.code.ei_reg.embeddings.google_word2vec import google_word2vec_model
from sources.code.ei_reg.random_forest.random_forest_deepmoji import predict_random_forest__deepmoji
from sources.code.ei_reg.svr.svr_deepmoji_and_features import predict_svr_deepmoji_and_features
from sources.code.ei_reg.svr.svr_deepmoji import predict_svr_deepmoji
from sources.code.ei_reg.attention_layer.google_word2vec_attention import google_word2vec_attention_model
from sources.code.ei_reg.attention_layer.glove_vectors_attention import glove_attention_model
from sources.code.predict.deepmoji_predict import predict_svr_deepmoji_live
from sources.code.ensemble import genetic_algorithm
from sources.loaders.loaders import parse_dataset
from sources.utils import ensemble_predictions, read_vectors_from_csv, run_lexicon_vectors, write_predictions, regression_to_ordinal
from sources.code.ei_oc.mapping import genetic_oc_algorithm

genetic_oc_algorithm('fear', 'dumps/EI-reg_en_fear_test_svr.txt')

# regression_to_ordinal('dumps/EI-reg_en_fear_test_svr.txt', 'fear', 'gold-no-mystery', [0.6, 0.7, 0.8, 1.0])
# glove_attention_model('anger')
# google_word2vec_attention_model('anger')
# genetic_algorithm('fear', ['dumps/EI-reg_en_fear_test_svr.txt', 'dumps/EI-reg_en_fear_test_random_forest.txt', 'dumps/EI-reg_en_fear_dev_google_attention_vectors.txt', 'dumps/EI-reg_en_fear_dev_glove_attention_vectors.txt'])
# print(run_lexicon_predictions('./datasets/EI-reg/training_set/arff/EI-reg-En-anger-train.arff'))
# print(map(float, read_vectors_from_csv('output.csv')[32][4:]))
# predict_random_forest__deepmoji('fear')
# my_word2vec_model('sadness')
# google_word2vec_model('fear')
# google_word2vec_attention_model('fear')

"""
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
"""
