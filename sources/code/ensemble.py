import numpy as np
import random
from scipy import stats
from deap import creator, base, tools, algorithms
from sources.loaders.files import find_path
from sources.utils import predictions_of_file, write_predictions_e_c, parse_dataset


def make_combined_predictions(weights, predictions):

    length = len(weights)
    weight_sum = sum(weights)
    final_predictions = []
    for prediction in range(len(predictions[0])):
        score = 0.0
        for line in range(length):
            score += predictions[line][prediction] * (weights[line] / sum(weights))
        final_predictions.append(score/weight_sum)
    return final_predictions


def genetic_algorithm(emotion, list_of_files, test_file):

    predictions = []
    for the_file in list_of_files:
        pr = predictions_of_file(the_file)
        predictions.append(pr)

    def get_pearson(weights):

        real_golden = predictions_of_file(find_path('EI-reg', emotion, test_file))
        predict_golden = make_combined_predictions(weights, predictions)
        if np.std(predict_golden) == 0 or np.std(predict_golden) == 0:
            return 0,
        return stats.pearsonr(predict_golden, real_golden)[0],

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, len(list_of_files))
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
        print(list(map(lambda x: x/sum(top), top)))
        print(get_pearson(top))

    x = [0.24647325800175182, 0.7377476334341374, 0.0, 0.015779108564110564]
    print(x)
    print(get_pearson(x))


def ensemble_e_c(list_of_files):
    scores = []
    predictions = []
    for my_file in list_of_files:
        with open(my_file, 'r') as fd:
            data = fd.readlines()
        data = [x.strip() for x in data][1:]
        data = [x.split('\t') for x in data]
        score = [[int(y) for y in x[2:]] for x in data]
        scores.append(score)
        fd.close()
    for i in range(len(scores[0])):
        new_pred = []
        for emotion in range(11):
            if sum([scores[j][i][emotion] for j in range(len(list_of_files))]) > 1:
                new_pred.append(1)
            else:
                new_pred.append(0)
        predictions.append(new_pred)
    print(scores[0][10])
    print(scores[1][10])
    print(scores[2][10])
    print(scores[3][10])
    print(scores[4][10])
    print(predictions[10])
    dev_dataset = parse_dataset('E-c', None, 'gold-no-mystery')
    write_predictions_e_c('dumps/ensemble_e_c', dev_dataset, predictions)
