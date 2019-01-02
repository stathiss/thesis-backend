import config


def find_file(task, emotion, label):
    """
    This function takes a task and an emotion and returns the file name.
    e.g:qd
    task = 'EI-reg',
    emotion = 'fear',
    label = 'train',
    input_file = 'EI-reg-En-fear-train.txt'
    """
    if label == 'development':
        input_file = '2018-' + task + '-En-' + emotion + '-dev' + '.txt'
    elif label == 'test':
        input_file = '2018-' + task + '-En-' + emotion + '-test' + '.txt'
    elif label == 'gold' or label == 'gold-no-mystery':
        input_file = '2018-' + task + '-En-' + emotion + '-test-' + label + '.txt'
    else:
        input_file = task + '-En-' + emotion + '-' + label + '.txt'

    return input_file


def find_path(task, emotion, label):
    """
    This function takes a task and an emotion and returns the path of file.
    e.g:qd
    task = 'EI-reg',
    emotion = 'fear',
    label = 'training'
    input_path = 'datasets/EI-reg/training_set/EI-reg-En-fear-train.txt'
    """
    if label == 'gold' or label == 'gold-no-mystery':
        input_path = config.DATA_DIR + '/gold-labels/' + task + '/' + find_file(task, emotion, label)
        return input_path
    elif label == 'train':
        label_ing = 'training'
    else:
        label_ing = label
    input_path = config.DATA_DIR + '/' + task + '/' + label_ing + '_set' + '/' + find_file(task, emotion, label)
    return input_path
