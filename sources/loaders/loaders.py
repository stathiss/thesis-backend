from sources.loaders.files import find_path


def parse_ei_reg(my_file):
    """
    :param my_file:
    :return: ids, tweets, emotion and score of EI-reg given file
    """
    with open(my_file, 'r') as fd:
        data = fd.readlines()
    data = [x.strip() for x in data][1:]
    data = [x.split('\t') for x in data]
    ids = [x[0] for x in data]
    tweet = [x[1] for x in data]
    emotion = [x[2] for x in data]
    score = [round(float(x[3]), 3) if x[3] != 'NONE' else None for x in data]
    fd.close()
    return ids, tweet, emotion, score


def parse_ei_oc(my_file):
    """
    :param my_file:
    :return: ids, tweets, emotion and score of EI-oc given file
    """
    with open(my_file, 'r') as fd:
        data = fd.readlines()
    data = [x.strip() for x in data][1:]
    data = [x.split('\t') for x in data]
    ids = [x[0] for x in data]
    tweet = [x[1] for x in data]
    emotion = [x[2] for x in data]
    score = [int(x[3].split(':')[0]) if x[3] != 'NONE' else None for x in data]
    fd.close()
    return ids, tweet, emotion, score


def parse_e_c(my_file):
    """
    :param my_file:
    :return: ids, tweets, emotion and score of E-c given file
    """
    with open(my_file, 'r') as fd:
        data = fd.readlines()
    data = [x.strip() for x in data][1:]
    data = [x.split('\t') for x in data]
    ids = [x[0] for x in data]
    tweet = [x[1] for x in data]
    print(data[0])
    score = [[int(x[2]), int(x[3]), int(x[4]), int(x[5]), int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11]), int(x[12])] if x[2] != 'NONE' else None for x in data]
    fd.close()
    return ids, tweet, None, score


def parse_dataset(task, emotion, label):
    my_file = find_path(task, emotion, label)
    if task == 'EI-reg':
        print(my_file)
        ids, tweet, emotion, score = parse_ei_reg(my_file)
    elif task == 'EI-oc':
        print(my_file)
        ids, tweet, emotion, score = parse_ei_oc(my_file)
    elif task == 'E-c':
        print(my_file)
        ids, tweet, emotion, score = parse_e_c(my_file)
    else:
        raise ValueError('Oopsie! It seems like you inserted something wrong')

    return ids, tweet, emotion, score
