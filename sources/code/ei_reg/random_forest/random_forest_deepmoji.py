import time
import pickle
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.loaders.loaders import parse_dataset
import random


def predict():

    X, y = deepmoji_vector('EI-reg', 'fear', 'train'), parse_dataset('EI-reg', 'fear', 'train')[3]
    with open('./dumps/deepmoji_eireg_fear_train', 'wb') as fp:
        pickle.dump(X, fp)
    clf = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    clf.fit(X, y)
    dev_dataset = parse_dataset('EI-reg', 'fear', 'development')
    test_input = deepmoji_vector('EI-reg', 'fear', 'development')
    with open('./dumps/deepmoji_eireg_fear_dev', 'wb') as fp:
        pickle.dump(test_input, fp)
    with open('./dumps/clf', 'wb') as fp:
        pickle.dump(clf, fp)
    prediction = clf.predict(test_input)
    outF = open("./dumps/EI-reg_en_fear_pred_dev.txt", "w")
    outF.write('ID\tTweet\tAffect\tDimension\tIntensity Score\n')
    for line in range(len(prediction)):
        # write line to output file
        outF.write(dev_dataset[0][line] + '\t' + dev_dataset[1][line] + '\t' + dev_dataset[2][line] + '\t' + str(prediction[line]))
        outF.write("\n")
    outF.close()

