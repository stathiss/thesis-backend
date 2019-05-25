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
        if not emotion:
            input_file = '2018-' + task + '-En-dev' + '.txt'
        else:
            input_file = '2018-' + task + '-En-' + emotion + '-dev' + '.txt'
    elif label == 'test':
        if not emotion:
            input_file = '2018-' + task + '-En-test' + '.txt'
        else:
            input_file = '2018-' + task + '-En-' + emotion + '-test' + '.txt'
    elif label == 'gold' or label == 'gold-no-mystery':
        if not emotion:
            input_file = '2018-E-c-En-test-gold.txt'
        else:
            input_file = '2018-' + task + '-En-' + emotion + '-test-' + label + '.txt'
    else:
        if not emotion:
            input_file = '2018-' + task + '-En-' + label + '.txt'
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
    if label == 'live':
        input_path = 'test_tweets.txt'
        return input_path
    elif label == 'gold' or label == 'gold-no-mystery':
        input_path = config.DATA_DIR + '/gold-labels/' + task + '/' + find_file(task, emotion, label)
        return input_path
    elif label == 'train':
        label_ing = 'training'
    else:
        label_ing = label
    input_path = config.DATA_DIR + '/' + task + '/' + label_ing + '_set' + '/' + find_file(task, emotion, label)
    return input_path
