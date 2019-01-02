import subprocess
import tensorflow as tf
from sources.loaders.loaders import parse_dataset
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


def get_pearson_correlation(task_type, prediction_file, gold_file):
    """
    task_type:
    1 for regression (EI-reg and V-reg tasks)
    2 for ordinal classification (EI-oc and V-oc tasks)
    3 for multi-label classification (E-c tasks).
    """
    output = subprocess.Popen(['python', 'sources/evaluation/evaluate.py', task_type, prediction_file, gold_file],
                              stdout=subprocess.PIPE).communicate()[0]
    total = float(output.split('\n')[0].split('\t')[1])
    range_half_to_one = float(output.split('\n')[1].split('\t')[1])
    return total, range_half_to_one


def write_predictions(file_name, dataset, prediction):
    """
    :param file_name: Input file
    :param dataset: Dataset (eg file of dev real values)
    :param prediction: Array of predictions you have to
    :return:
    """

    out_file = open(file_name, "w")
    out_file.write('ID\tTweet\tAffect\tDimension\tIntensity Score\n')

    for line in range(len(prediction)):
        # write line to output file
        out_file.write(dataset[0][line] + '\t' + dataset[1][line] + '\t'
                   + dataset[2][line] + '\t' + str(prediction[line]))
        out_file.write("\n")
    out_file.close()


def predictions_of_file(my_file):
    with open(my_file, 'r') as fd:
        data = fd.readlines()
    data = [x.strip() for x in data][1:]
    data = [x.split('\t') for x in data]
    score = [float(x[3]) for x in data]
    fd.close()
    return score


def ensemble_weights(my_list, index, powerset):
    if index == len(my_list) -1 and my_list[index] == 1.0:
        powerset.append(my_list)
        return powerset
    elif my_list[index] == 1.0:
        powerset.append()
        my_list[index] == 0.0
        index += 1
    return powerset


def ensemble_predictions(files, weights, task, emotion, label, gold_file):
    # check if weights sum up to 1
    weights_score = 0
    for weight in weights:
        weights_score += weight
    if not 0.99 < weights_score < 1.01:
        raise ValueError('Oops... Seems like you entered the weights wrong!')

    dataset = parse_dataset(task, emotion, label)
    predictions = []
    length = len(files)
    for temp_file in files:
        predictions.append(predictions_of_file(temp_file))
    final_predictions = []
    for prediction in range(len(predictions[0])):
        score = 0.0
        for line in range(length):
            score += predictions[line][prediction]*weights[line]
        final_predictions.append(score)
    print(len(final_predictions))
    write_predictions('dumps/combine_predictions.txt', dataset, final_predictions)
    print(get_pearson_correlation('1', 'dumps/combine_predictions.txt', gold_file))


def pearson_correlation_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1-r


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        self.W = None
        self.b = None
        self.built = None
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, _input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
