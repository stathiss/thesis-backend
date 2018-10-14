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
    if label == 'train':
        label_ing = 'training'
    else:
        label_ing = label
    input_path = config.DATA_DIR + '/' + task + '/' + label_ing + '_set' + '/' + find_file(task, emotion, label)
    return input_path
