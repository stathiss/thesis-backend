from __future__ import division

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sources.features.tweet_specific_features.tweet_specific_features import parse_tweet_specific_features
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.loaders.loaders import parse_dataset
from sources.loaders.files import find_path
from sources.utils import get_pearson_correlation, write_predictions, run_lexicon_vectors, read_vectors_from_csv,\
    normalize_vectors


def predict_random_forest_deepmoji_and_features(emotion, add_lexicons=False, add_features=False):
    bootstrap = True
    max_depth = 10
    max_features = 'sqrt'
    min_samples_leaf = 4
    min_samples_split = 2
    n_estimators = 600
    train_file = 'train_and_dev'
    test_file = 'gold-no-mystery'

    # Load datasets
    X = deepmoji_vector('EI-reg', emotion, train_file)
    y = parse_dataset('EI-reg', emotion, train_file)[3]

    test_input = deepmoji_vector('EI-reg', emotion, test_file)

    # Append all of them
    if add_lexicons:
        # Load weka lexicon features
        run_lexicon_vectors('./datasets/EI-reg/train_and_dev_set/arff/EI-reg-En-' + str(emotion) + '-train-dev.arff')
        X_lexicon_extension = read_vectors_from_csv('output.csv')
        print('xxxxx', len(X_lexicon_extension))
        X_lexicon_extension_variables = normalize_vectors(X_lexicon_extension)[1]
        X_lexicon_extension = normalize_vectors(X_lexicon_extension)[0]
        run_lexicon_vectors('./datasets/EI-reg/test_set/arff/2018-EI-reg-En-' + str(emotion) + '-gold.arff')
        test_input_lexicon_extension = read_vectors_from_csv('output.csv')
        for i in range(len(test_input_lexicon_extension[0])):
            for line in test_input_lexicon_extension:
                line[i] = (line[i] - X_lexicon_extension_variables[i][0]) / X_lexicon_extension_variables[i][1]
        print(len(X), len(X_lexicon_extension))
        X = np.hstack([X, X_lexicon_extension])
        test_input = np.hstack([test_input, test_input_lexicon_extension])

    if add_features:
        # Load tweet specific features
        X_features_extension = parse_tweet_specific_features('EI-reg', emotion, train_file)
        test_input_features_extension = parse_tweet_specific_features('EI-reg', emotion, test_file)
        print(X_features_extension[0])

        X = np.hstack([X, X_features_extension])
        test_input = np.hstack(([test_input, test_input_features_extension]))
    print(X[0])
    dev_dataset = parse_dataset('EI-reg', emotion, test_file)

    clf = RandomForestRegressor(bootstrap=bootstrap,
                                max_depth=max_depth,
                                max_features=max_features,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                n_estimators=n_estimators)
    clf.fit(X, y)
    predictions = clf.predict(test_input)
    file_name = "./dumps/EI-reg/" + test_file + "/DeepMoji/EI-reg_en_" + emotion + (
        "_lexicons" if add_lexicons else "") + ("_features" if add_features else "") + "_random_forest.txt"
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation('1', file_name, find_path('EI-reg', emotion, 'gold-no-mystery')))
