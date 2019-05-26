import numpy as np
import pickle
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector


def predict_svr_deepmoji_live():
    test_input = deepmoji_vector('EI-reg', 'emotion', 'live')

    with open('dumps/svr_deepmoji_anger', 'rb') as fp:
        clf_anger = pickle.load(fp)
    predictions_anger = clf_anger.predict(test_input)

    with open('dumps/svr_deepmoji_fear', 'rb') as fp:
        clf_fear = pickle.load(fp)
    predictions_fear = clf_fear.predict(test_input)

    with open('dumps/svr_deepmoji_joy', 'rb') as fp:
        clf_joy = pickle.load(fp)
    predictions_joy = clf_joy.predict(test_input)

    with open('dumps/svr_deepmoji_sadness', 'rb') as fp:
        clf_sadness = pickle.load(fp)
    predictions_sadness = clf_sadness.predict(test_input)

    with open('dumps/svm_model', 'rb') as fp:
        clf_e_c = pickle.load(fp)
    predictions_e_c = clf_e_c.predict(test_input)

    return np.round(predictions_anger, 2),\
        np.round(predictions_fear, 2),\
        np.round(predictions_joy, 2),\
        np.round(predictions_sadness, 2),\
        predictions_e_c
