from sklearn.ensemble import RandomForestRegressor
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.loaders.loaders import parse_dataset
from sources.loaders.files import find_path
from sources.utils import get_pearson_correlation, write_predictions


def predict_random_forest_deepmoji(emotion):
    train_file = 'train_and_dev'
    test_file = 'gold-no-mystery'

    bootstrap = True
    max_depth = 10
    max_features = 'sqrt'
    min_samples_leaf = 4
    min_samples_split = 2
    n_estimators = 600

    X = deepmoji_vector('EI-reg', emotion, train_file)
    y = parse_dataset('EI-reg', emotion, train_file)[3]

    test_input = deepmoji_vector('EI-reg', emotion, test_file)
    dev_dataset = parse_dataset('EI-reg', emotion, test_file)
    clf = RandomForestRegressor(bootstrap=bootstrap,
                                max_depth=max_depth,
                                max_features=max_features,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                n_estimators=n_estimators)
    clf.fit(X, y)
    predictions = clf.predict(test_input)
    file_name = "./dumps/EI-reg/" + test_file + "/DeepMoji/EI-reg_en_" + emotion + "_random_forest.txt"
    write_predictions(file_name, dev_dataset, predictions)
    print(file_name)
    print(get_pearson_correlation('1', file_name, find_path('EI-reg', emotion, test_file)))
