from sources.code.ei_reg.embeddings.my_word2vec_lstm import my_word2vec_model
from sources.code.ei_reg.attention_layer.my_word2vec_attention import my_word2vec_attention_model
from sources.code.ei_reg.embeddings.glove_vectors import glove_model
from sources.code.ei_reg.embeddings.google_word2vec import google_word2vec_model
from sources.code.ei_reg.random_forest.random_forest_deepmoji import predict_random_forest_deepmoji
from sources.code.ei_reg.random_forest.random_forest_deepmoji_and_features import predict_random_forest_deepmoji_and_features
from sources.code.ei_reg.svr.svr_deepmoji_and_features import predict_svr_deepmoji_and_features
from sources.code.ei_reg.svr.svr_deepmoji import predict_svr_deepmoji
from sources.code.ei_reg.attention_layer.google_word2vec_attention import google_word2vec_attention_model
from sources.code.ei_reg.attention_layer.glove_vectors_attention import glove_attention_model
from sources.code.predict.deepmoji_predict import predict_svr_deepmoji_live
from sources.code.ensemble import genetic_algorithm, ensemble_e_c
from sources.loaders.loaders import parse_dataset
from sources.code.e_c.svm import predict_svm_deepmoji_and_features, write_svm
from sources.code.e_c.random_forest import predict_rf_deepmoji_and_features, write_rf
from sources.utils import ensemble_predictions, read_vectors_from_csv, run_lexicon_vectors, write_predictions, regression_to_ordinal, get_second_biggest_index
from sources.code.ei_oc.mapping import genetic_oc_algorithm

# genetic_oc_algorithm('fear', 'dumps/EI-reg_en_fear_test_svr.txt')
"""
glove_attention_model('anger', 50)
glove_attention_model('fear', 50)
glove_attention_model('joy', 50)
glove_attention_model('sadness', 50)

glove_attention_model('anger', 100)
glove_attention_model('fear', 100)
glove_attention_model('joy', 100)
glove_attention_model('sadness', 100)

glove_attention_model('anger', 200)
glove_attention_model('fear', 200)

glove_attention_model('joy', 200)
glove_attention_model('sadness', 200)

glove_attention_model('anger', 300)
glove_attention_model('fear', 300)
glove_attention_model('joy', 300)
glove_attention_model('sadness', 300)
"""

"""
google_word2vec_attention_model('anger')
google_word2vec_attention_model('fear')
google_word2vec_attention_model('joy')
google_word2vec_attention_model('sadness')

my_word2vec_attention_model('anger')
my_word2vec_attention_model('fear')
my_word2vec_attention_model('joy')
my_word2vec_attention_model('sadness')
"""
# predict_rf_deepmoji_and_features()
# write_rf()
# predict_rf_deepmoji_and_features(add_lexicons=True)
# predict_rf_deepmoji_and_features(add_features=True)
# predict_rf_deepmoji_and_features(add_lexicons=True, add_features=True)
ensemble_e_c(['./dumps/E-c/gold-no-mystery/DeepMoji/E-c_en_lexicons_features_svm.txt',
              './dumps/E-c/gold-no-mystery/DeepMoji/E-c_en_features_svm.txt',
              './dumps/E-c/gold-no-mystery/DeepMoji/E-c_en_lexicons_svm.txt',
              './dumps/E-c/gold-no-mystery/DeepMoji/E-c_en_lexicons_features_rf.txt',
              './dumps/E-c/gold-no-mystery/DeepMoji/E-c_en_lexicons_rf.txt'])

"""


glove_model('anger', 50)
glove_model('fear', 50)
glove_model('joy', 50)
glove_model('sadness', 50)

glove_model('anger', 100)
glove_model('fear', 100)
glove_model('joy', 100)
glove_model('sadness', 100)

glove_model('anger', 200)
glove_model('fear', 200)
glove_model('joy', 200)
glove_model('sadness', 200)

glove_model('anger', 300)
glove_model('fear', 300)
glove_model('joy', 300)
glove_model('sadness', 300)

google_word2vec_model('anger')
google_word2vec_model('fear')
google_word2vec_model('joy')
google_word2vec_model('sadness')

my_word2vec_model('anger')
my_word2vec_model('fear')
my_word2vec_model('joy')
my_word2vec_model('sadness')

glove_attention_model('fear', 300)
glove_attention_model('joy', 300)
glove_attention_model('sadness', 300)


my_word2vec_attention_model('anger')
my_word2vec_attention_model('fear')
my_word2vec_attention_model('joy')
my_word2vec_attention_model('sadness')

predict_random_forest_deepmoji_and_features('anger', add_features=True)
predict_random_forest_deepmoji_and_features('fear', add_features=True)
predict_random_forest_deepmoji_and_features('joy', add_features=True)
predict_random_forest_deepmoji_and_features('sadness', add_features=True)


predict_svr_deepmoji_and_features('anger', add_features=True)
predict_svr_deepmoji_and_features('fear', add_features=True)
predict_svr_deepmoji_and_features('joy', add_features=True)
predict_svr_deepmoji_and_features('sadness', add_features=True)

predict_svr_deepmoji_and_features('anger', add_lexicons=True, add_features=True)
predict_svr_deepmoji_and_features('fear', add_lexicons=True, add_features=True)
predict_svr_deepmoji_and_features('joy', add_lexicons=True, add_features=True)
predict_svr_deepmoji_and_features('sadness', add_lexicons=True, add_features=True)



google_word2vec_model('anger')
google_word2vec_model('fear')
google_word2vec_model('joy')
google_word2vec_model('sadness')

glove_model('anger', 50)
glove_model('fear', 50)
glove_model('joy', 50)
glove_model('sadness', 50)

glove_model('anger', 100)
glove_model('fear', 100)
glove_model('joy', 100)
glove_model('sadness', 100)

glove_model('anger', 200)
glove_model('fear', 200)
glove_model('joy', 200)
glove_model('sadness', 200)

glove_model('anger', 300)
glove_model('fear', 300)
glove_model('joy', 300)
glove_model('sadness', 300)

google_word2vec_model('anger')
google_word2vec_model('fear')
google_word2vec_model('joy')
google_word2vec_model('sadness')
"""

# regression_to_ordinal('dumps/EI-reg_en_fear_test_svr.txt', 'fear', 'gold-no-mystery', [0.6, 0.7, 0.8, 1.0])
# glove_attention_model('anger')
# google_word2vec_attention_model('anger')
# genetic_algorithm('fear', ['dumps/EI-reg//gold-no-mystery/DeepMoji/EI-reg_en_joy_features_svr.txt', 'dumps/EI-reg//gold-no-mystery/DeepMoji/EI-reg_en_sadness_lexicons_random_forest.txt', 'dumps/EI-reg//gold-no-mystery/BiLSTM+Att/EI-reg_en_sadness_glove_300.txt', 'dumps/EI-reg//gold-no-mystery/BiLSTM+Att/EI-reg_en_sadness_google.txt'])
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
