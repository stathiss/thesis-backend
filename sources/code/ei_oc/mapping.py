import numpy as np
import random
from scipy import stats
from deap import creator, base, tools, algorithms
from sources.loaders.files import find_path
from sources.utils import predictions_of_file, write_predictions, get_pearson_correlation
from sources.loaders.loaders import parse_dataset


def make_oc_predictions(weights, predictions, emotion):

    max_weight = max(weights)
    weights = sorted(weights)
    for w in range(len(weights)):
        weights[w] = weights[w] / max_weight
    ordinal_c = []
    for pr in predictions:
        if pr < weights[0]:
            ordinal_c.append('0: no ' + emotion + ' can be inferred')
        elif pr < weights[1]:
            ordinal_c.append('1: low amount of ' + emotion + ' can be inferred')
        elif pr < weights[2]:
            ordinal_c.append('2: moderate amount of ' + emotion + ' can be inferred')
        else:
            ordinal_c.append('3: high amount of ' + emotion + ' can be inferred')
    return ordinal_c


def genetic_oc_algorithm(emotion, the_file):

    predictions = predictions_of_file(the_file)

    def get_pearson(weights):

        dev_dataset = parse_dataset('EI-oc', emotion, 'gold-no-mystery')
        predict_golden = make_oc_predictions(weights, predictions, emotion)
        if np.std(predict_golden) == 0 or np.std(predict_golden) == 0 or sum(weights) >= 1:
            return 0,
        file_name = "./dumps/test_oc.txt"
        write_predictions(file_name, dev_dataset, predict_golden)
        print(get_pearson_correlation('2', file_name, find_path('EI-oc', emotion, 'gold-no-mystery'))[0])
        return get_pearson_correlation('2', file_name, find_path('EI-oc', emotion, 'gold-no-mystery'))[0],

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", get_pearson)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=200)

    generations = 40
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    top10 = tools.selBest(population, k=10)

    for top in top10:
        print(top)
        print(get_pearson(top))
