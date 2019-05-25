import pickle
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.features.tweet_specific_features.tweet_specific_features import parse_tweet_specific_features
from sources.loaders.loaders import parse_dataset
from sources.loaders.files import find_path
from sources.utils import get_pearson_correlation, write_predictions_e_c, run_lexicon_vectors, read_vectors_from_csv, \
    normalize_vectors
import numpy as np


def predict_svm_deepmoji_and_features(add_lexicons=False, add_features=False):
    emotion = None
    train_file = 'train_and_dev'
    test_file = 'gold-no-mystery'
    # Load datasets

    # X = deepmoji_vector('E-c', emotion, train_file)
    with open('dumps/svm_e_c_x', 'rb') as fp:
        X = pickle.load(fp)

    # y = parse_dataset('E-c', emotion, train_file)[3]
    with open('dumps/svm_e_c_y', 'rb') as fp:
        y = pickle.load(fp)

    # test_input = deepmoji_vector('E-c', emotion, test_file)
    with open('dumps/svm_e_c_test_input', 'rb') as fp:
        test_input = pickle.load(fp)

    # Append all of them
    if add_lexicons:
        # Load weka lexicon features
        run_lexicon_vectors('./datasets/E-c/train_and_dev_set/arff/2018-E-c-En-train-dev.arff')
        X_lexicon_extension = read_vectors_from_csv('output.csv')
        print('xxxxx', len(X_lexicon_extension))

        X_lexicon_extension_variables = normalize_vectors(X_lexicon_extension)[1]
        X_lexicon_extension = normalize_vectors(X_lexicon_extension)[0]
        print('ela', X_lexicon_extension_variables)
        run_lexicon_vectors('./datasets/E-c/test_set/arff/2018-E-c-En-gold.arff')
        test_input_lexicon_extension = read_vectors_from_csv('output.csv')
        for i in range(len(test_input_lexicon_extension[0])):
            for line in test_input_lexicon_extension:
                line[i] = (line[i] - X_lexicon_extension_variables[i][0]) / X_lexicon_extension_variables[i][1]
        print(len(X[0]), len(X_lexicon_extension[0]))
        X = np.hstack([X, X_lexicon_extension])
        test_input = np.hstack([test_input, test_input_lexicon_extension])

    if add_features:
        # Load tweet specific features
        X_features_extension = parse_tweet_specific_features('E-c', emotion, train_file)
        test_input_features_extension = parse_tweet_specific_features('E-c', emotion, test_file)
        print(X_features_extension[0])

        X = np.hstack([X, X_features_extension])
        test_input = np.hstack(([test_input, test_input_features_extension]))
    print('elo', X[0])
    dev_dataset = parse_dataset('E-c', emotion, test_file)

    clf = MultiOutputClassifier(LinearSVC(
        C=1.0, class_weight=None, dual=True, fit_intercept=True,
        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
        multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=1), n_jobs=-1)
    clf.fit(np.array(X), np.array(y))
    with open('dumps/svm_model' + ("_lexicons" if add_lexicons else "") + ("_features" if add_features else ""), 'wb') as fp:
        pickle.dump(clf, fp)

    predictions = clf.predict(np.array(test_input))
    with open('dumps/svm_predictions' + ("_lexicons" if add_lexicons else "") + ("_features" if add_features else ""), 'wb') as fp:
        pickle.dump(predictions, fp)

    file_name = "./dumps/E-c/" + test_file + "/DeepMoji/E-c_en" + ("_lexicons" if add_lexicons else "") + ("_features" if add_features else "") + "_svm.txt"
    write_predictions_e_c(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation('5', file_name, find_path('E-c', emotion, test_file)))


def write_svm():
    with open('dumps/svm_predictions', 'rb') as fp:
        predictions = pickle.load(fp)

    print(predictions[:10])
    file_name = "./dumps/test_svm.txt"
    dev_dataset = parse_dataset('E-c', None, 'gold-no-mystery')
    write_predictions_e_c(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation('5', file_name, find_path('E-c', None, 'gold-no-mystery')))
