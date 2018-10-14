from sources.loaders.files import find_path


def parse_ei_reg(file):
    """
    :param file:
    :return: ids, tweets, emotion and score of EI-reg given file
    """
    with open(file, 'r') as fd:
        data = fd.readlines()
    data = [x.strip() for x in data][1:]
    data = [x.split('\t') for x in data]
    ids = [x[0] for x in data]
    tweet = [x[1] for x in data]
    emotion = [x[2] for x in data]
    score = [x[3] for x in data]
    fd.close()
    return ids, tweet, emotion, score


def parse_ei_oc(file):
    """
    :param file:
    :return: ids, tweets, emotion and score of EI-oc given file
    """
    ids = []
    tweet = []
    emotion = []
    score = []
    return ids, tweet, emotion, score


def parse_e_c(file):
    """
    :param file:
    :return: ids, tweets, emotion and score of E-c given file
    """
    ids = []
    tweet = []
    emotion = []
    score = []
    return ids, tweet, emotion, score


def parse_dataset(task, emotion, label):
    file = find_path(task, emotion, label)
    if task == 'EI-reg':
        print(file)
        ids, tweet, emotion, score = parse_ei_reg(file)
    elif task == 'EI-oc':
        ids, tweet, emotion, score = parse_ei_oc(file)
    elif task == 'E-c':
        ids, tweet, emotion, score = parse_e_c(file)
    else:
        raise ValueError('Oopsie! It seems like you inserted something wrong')

    return ids, tweet, emotion, score
