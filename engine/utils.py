import numpy as np


def aging(population_fitness, age_distribution):
    """
    This function takes a population's population_fitness and age distribution, and
    applies a scaling factor based on age. The scaling factors are:
    - 0.5 for ages below 0.25 (young)
    - 1.5 for ages between 0.25 and 0.75 (adult)
    - 0.5 for ages above 0.75 (old)
    """
    young_scale, adult_scale, old_scale = 0.5, 1.5, 0.5
    thresholds = [0.25, 0.75]
    scales = [young_scale, adult_scale, old_scale]

    return [fitness * scales[sum(age > threshold for threshold in thresholds)]
            for age, fitness in zip(age_distribution, population_fitness)]


def calculate_selection_pressure_fitness_variance(population_fitness):
    """
    Calculate the selection pressure (exploitation factor) using fitness variance.
    """
    population_fitness_copy = np.array(population_fitness)
    return np.var(population_fitness_copy) / np.mean(population_fitness_copy)


def calculate_selection_pressure_top_average_selection(population_fitness):
    """
    Calculate the selection pressure (exploitation factor) using top-average selection probability ratio.
    """
    population_fitness_copy = np.array(population_fitness)
    population_fitness_copy.sort()
    top_10_percent = int(0.1 * len(population_fitness_copy))
    top_individuals = population_fitness_copy[-top_10_percent:]
    return np.mean(top_individuals) / np.mean(population_fitness_copy)

