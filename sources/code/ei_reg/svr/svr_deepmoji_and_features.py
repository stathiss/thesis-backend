from sklearn.svm import SVR
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.features.tweet_specific_features.tweet_specific_features import parse_tweet_specific_features
from sources.loaders.loaders import parse_dataset
from sources.loaders.files import find_path
from sources.utils import get_pearson_correlation, write_predictions, run_lexicon_vectors, read_vectors_from_csv,\
    normalize_vectors
import numpy as np


def predict_svr_deepmoji_and_features(emotion, add_lexicons=False, add_features=False):
    # Load datasets
    X = deepmoji_vector('EI-reg', emotion, 'train-and-dev')
    y = parse_dataset('EI-reg', emotion, 'train-and-dev')[3]

    test_input = deepmoji_vector('EI-reg', emotion, 'gold-no-mystery')

    # Append all of them
    if add_lexicons:
        # Load weka lexicon features
        run_lexicon_vectors('./datasets/EI-reg/training_set/arff/EI-reg-En-' + str(emotion) + '-train.arff')
        X_lexicon_extension = read_vectors_from_csv('output.csv')
        X_lexicon_extension_variables = normalize_vectors(X_lexicon_extension)[1]
        X_lexicon_extension = normalize_vectors(X_lexicon_extension)[0]
        run_lexicon_vectors('./datasets/EI-reg/development_set/arff/2018-EI-reg-En-' + str(emotion) + '-dev.arff')
        test_input_lexicon_extension = read_vectors_from_csv('output.csv')
        for i in range(len(test_input_lexicon_extension[0])):
            for line in test_input_lexicon_extension:
                line[i] = (line[i] - X_lexicon_extension_variables[i][0]) / X_lexicon_extension_variables[i][1]
        X = np.hstack([X, X_lexicon_extension])
        test_input = np.hstack([test_input, test_input_lexicon_extension])

    if add_features:
        # Load tweet specific features
        X_features_extension = parse_tweet_specific_features('EI-reg', emotion, 'train')
        test_input_features_extension = parse_tweet_specific_features('EI-reg', emotion, 'development')
        print(X_features_extension[0])

        X = np.hstack([X, X_features_extension])
        test_input = np.hstack(([test_input, test_input_features_extension]))
    print(X[0])
    dev_dataset = parse_dataset('EI-reg', emotion, 'development')

    clf = SVR(kernel='rbf', C=10, gamma=0.0001, epsilon=0.05)
    clf.fit(X, y)
    predictions = clf.predict(test_input)

    file_name = "./dumps/EI-reg_en_" + emotion + "_test_svr.txt"
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation('1', file_name, find_path('EI-reg', emotion, 'development')))
