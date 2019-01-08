from sklearn.svm import SVR
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.loaders.loaders import parse_dataset
from sources.utils import get_pearson_correlation, write_predictions, run_lexicon_vectors, read_vectors_from_csv


def predict_svr_deepmoji_and_lexicons(emotion):
    X = deepmoji_vector('EI-reg', emotion, 'train')
    y = parse_dataset('EI-reg', emotion, 'train')[3]
    test_input = deepmoji_vector('EI-reg', emotion, 'development')
    run_lexicon_vectors('./datasets/EI-reg/training_set/arff/EI-reg-En-' + str(emotion) + '-train.arff')
    # X_extension = read_vectors_from_csv('output.csv')
    dev_dataset = parse_dataset('EI-reg', emotion, 'development')
    clf = SVR(kernel='rbf', C=10, gamma=0.0001, epsilon=0.05)
    clf.fit(X, y)
    predictions = clf.predict(test_input)
    file_name = "./dumps/EI-reg_en_" + emotion + "_test_svr.txt"
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation(
        '1',
        file_name,
        'datasets/gold-labels/EI-reg/2018-EI-reg-En-' + emotion + '-test-gold-no-mystery.txt'))
