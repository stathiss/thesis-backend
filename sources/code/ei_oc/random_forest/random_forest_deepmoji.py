from sklearn.ensemble import RandomForestRegressor
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.loaders.loaders import parse_dataset
from sources.utils import get_pearson_correlation, write_predictions
import pickle


def predict(emotion):
    bootstrap = True
    max_depth = 10
    max_features = 'sqrt'
    min_samples_leaf = 4
    min_samples_split = 2
    n_estimators = 600

    X = deepmoji_vector('EI-reg', emotion, 'train')
    with open('dumps/deepmoji_vector_' + emotion + '_train', 'wb') as fp:
        pickle.dump(X, fp)
    y = parse_dataset('EI-reg', emotion, 'train')[3]

    test_input = deepmoji_vector('EI-reg', emotion, 'development')
    with open('dumps/deepmoji_vector_' + emotion + '_dev', 'wb') as fp:
        pickle.dump(test_input, fp)
    dev_dataset = parse_dataset('EI-reg', emotion, 'development')
    max_score = 0
    max_variables = ''
    clf = RandomForestRegressor(bootstrap=bootstrap,
                                max_depth=max_depth,
                                max_features=max_features,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                n_estimators=n_estimators)
    clf.fit(X, y)
    prediction = clf.predict(test_input)
    file_name = './dumps/EI-reg_en_' + emotion + '_pred_dev_random_forest_' +\
                str(max_depth) + '_' +\
                str(max_features) + '_' +\
                str(min_samples_leaf) + '_' +\
                str(min_samples_split) + '_' +\
                str(n_estimators) + '_' + '.txt'
    write_predictions(file_name, dev_dataset, prediction)
    print(file_name)
    print(get_pearson_correlation('1', file_name,
                                  'datasets/EI-reg/development_set/2018-EI-reg-En-' + emotion + '-dev.txt'))
    print(' ')
    if get_pearson_correlation('1', file_name,
                               'datasets/EI-reg/development_set/2018-EI-reg-En-' + emotion + '-dev.txt')[0] > max_score:
        max_score = get_pearson_correlation('1', file_name,
                                            'datasets/EI-reg/development_set/2018-EI-reg-En-' + emotion + '-dev.txt')[0]
        max_variables = file_name
    print('MAAAAAAAAAAAAAAAXXXX')
    print(max_score)
    print(max_variables)
