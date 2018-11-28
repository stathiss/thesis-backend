from sklearn.svm import SVR
from sources.features.deepmoji_feature.deepmoji_vector import deepmoji_vector
from sources.loaders.loaders import parse_dataset


def predict():
    X = deepmoji_vector('EI-reg', 'sadness', 'train')
    y = parse_dataset('EI-reg', 'sadness', 'train')[3]
    test_input = deepmoji_vector('EI-reg', 'sadness', 'development')
    dev_dataset = parse_dataset('EI-reg', 'sadness', 'development')
    clf = SVR(kernel='rbf', C=10, gamma=0.0001, epsilon=0.05)
    clf.fit(X, y)
    prediction = clf.predict(test_input)
    outF = open("./dumps/EI-reg_en_sadness_pred_dev.txt", "w")
    outF.write('ID\tTweet\tAffect\tDimension\tIntensity Score\n')
    for line in range(len(prediction)):
        # write line to output file
        outF.write(dev_dataset[0][line] + '\t' + dev_dataset[1][line] + '\t'
                   + dev_dataset[2][line] + '\t' + str(prediction[line]))
        outF.write("\n")
    outF.close()
