import pickle
from sklearn.svm import SVR
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.loaders.loaders import parse_dataset
from sources.loaders.files import find_path
from sources.utils import get_pearson_correlation, write_predictions


def predict_svr_deepmoji(emotion):
    X = deepmoji_vector('EI-reg', emotion, 'train_and_dev')
    y = parse_dataset('EI-reg', emotion, 'train_and_dev')[3]
    test_input = deepmoji_vector('EI-reg', emotion, 'gold-no-mystery')
    dev_dataset = parse_dataset('EI-reg', emotion, 'gold-no-mystery')
    clf = SVR(kernel='rbf', C=10, gamma=0.0001, epsilon=0.05)
    clf.fit(X, y)
    with open('dumps/svr_deepmoji_' + emotion, 'wb') as fp:
        pickle.dump(clf, fp)
    predictions = clf.predict(test_input)
    file_name = "./dumps/EI-reg_en_" + emotion + "_test_svr.txt"
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation('1', file_name, find_path('EI-reg', emotion, 'gold-no-mystery')))
