import numpy as np
import random
from scipy import stats
from deap import creator, base, tools, algorithms
from sources.loaders.files import find_path
from sources.utils import predictions_of_file


def make_combined_predictions(weights, predictions):

    length = len(weights)
    weight_sum = sum(weights)
    final_predictions = []
    for prediction in range(len(predictions[0])):
        score = 0.0
        for line in range(length):
            score += predictions[line][prediction] * weights[line]
        final_predictions.append(score/weight_sum)
    return final_predictions


def genetic_algorithm(emotion, list_of_files):

    predictions = []
    for the_file in list_of_files:
        pr = predictions_of_file(the_file)
        predictions.append(pr)

    def get_pearson(weights):

        golden = predictions_of_file(find_path('EI-reg', emotion, 'gold-no-mystery'))
        predict_golden = make_combined_predictions(weights, predictions)
        if np.std(predict_golden) == 0 or np.std(predict_golden) == 0 or sum(weights) >= 1:
            return 0,
        return stats.pearsonr(predict_golden, golden)[0],

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
        print(top)
        print(get_pearson(top))

    golden = predictions_of_file(find_path('EI-reg', emotion, 'gold-no-mystery'))

    print(stats.pearsonr(predictions[0], golden)[0])
    print(stats.pearsonr(predictions[1], golden)[0])
    print(stats.pearsonr(make_combined_predictions([0.3, 0.7, 0.1, 0.1], predictions), golden)[0])
