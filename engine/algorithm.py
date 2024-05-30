import dataclasses
import random
import time
import typing
from enum import Enum

import numpy as np
from tqdm import tqdm

from engine.selection import rank, rws, sus, tournament
from sudoku.utils import aging


class SelectionMethod(Enum):
    SUS = 1
    RWS = 2
    TOURNAMENT = 3
    RANK = 4


@dataclasses.dataclass
class GeneticSettings:
    """Configuration for the genetic algorithm."""
    use_aging: bool
    genes_count: int
    population_size: int
    max_generations: int
    mutation_rate: float
    selection: SelectionMethod
    mutation_generator: typing.Callable[[any, float], any]
    crossover_generator: typing.Callable[[any, any], any]
    fitness_calculator: typing.Callable[[any], float]
    individual_generator: typing.Callable[[int], any]
    elite_size: float = 0.1
    verbose: bool = True
    print_function: typing.Callable = print


def run_genetic_algorithm(settings: GeneticSettings):
    all_generations = []
    all_fitness_scores = []
    start_cpu_time = time.process_time()

    # Initialize the ages of the population
    ages = [random.random() for _ in range(settings.population_size)]

    # Generate the initial population
    population = [settings.individual_generator(settings.genes_count) for _ in range(settings.population_size)]
    print(population)

    # Check population sizes
    for individual in population:
        if len(individual) != settings.genes_count:
            print(f"Error: Individual size mismatch. Expected {settings.genes_count}, got {len(individual)}")
            return

    # Run the genetic algorithm
    for generation in tqdm(range(settings.max_generations), desc="Genetic Algorithm Progress"):
        if settings.verbose:
            print(f"Generation {generation}")
        generation_cpu_start = time.process_time()

        # Calculate the fitness of the population
        population_fitness = [settings.fitness_calculator(individual) for individual in population]

        # Store the fitness scores and generation number
        all_fitness_scores.append(population_fitness)
        all_generations.append(generation)

        # Print the average fitness and standard deviation of the population
        avg = np.mean(population_fitness)
        dev = np.std(population_fitness)

        if settings.verbose:
            print("--------------------")
            print(
                f"Generation {generation}: "
                f"Average Fitness = {int(round(avg))}, "
                f"Selection Pressure Exploitation Factor: "
                f"Fitness Variance = {calculate_selection_pressure_fitness_variance(population_fitness)}, "
                f"Top Average Selection = {calculate_selection_pressure_top_average_selection(population_fitness)} "
                f"Max fitness = {max(population_fitness)}"
            )
            print("--------------------")

        # Check for convergence
        elite_size = int(settings.population_size * settings.elite_size)
        elite_indices = sorted(
            range(settings.population_size), key=lambda i: population_fitness[i], reverse=True
        )[:elite_size]

        # Store the elite individuals
        offspring = []
        elites = [population[i] for i in elite_indices]
        if settings.use_aging:
            population_fitness = aging(population_fitness, ages)

        while len(offspring) < settings.population_size - elite_size:
            parent1, parent2 = get_parents(settings.selection, population, population_fitness, elites)

            crossover_result = settings.crossover_generator(parent1, parent2)
            if isinstance(crossover_result, list) or isinstance(crossover_result, tuple):
                children = crossover_result
            else:
                children = [crossover_result]

            for child in children:
                child = settings.mutation_generator(child, settings.mutation_rate)
                offspring.append(child)

        population = elites + offspring
        ages = [age + 1 / settings.population_size for age in ages]

        # Increment the ages of the population
        generation_cpu_end = time.process_time()
        generation_cpu_ticks = generation_cpu_end - generation_cpu_start
        generation_cpu_elapsed = generation_cpu_end - start_cpu_time
        if settings.verbose:
            print(f"cpu Generation {generation}: Ticks Clock cpu = {generation_cpu_ticks} seconds, "
                  f"Total Elapsed = {generation_cpu_elapsed:.2f} seconds")

    cpu_convergence = time.process_time() - start_cpu_time
    if settings.verbose:
        print(f"cpu Time to convergence: {cpu_convergence:.2f} seconds")

    best_individual = max(population, key=lambda i: settings.fitness_calculator(i))
    best_fitness = settings.fitness_calculator(best_individual)
    return best_individual, best_fitness, all_fitness_scores, all_generations


def get_parents(selection, population, population_fitness, elites=None):
    choices: dict[SelectionMethod, callable] = {
        SelectionMethod.SUS: sus,
        SelectionMethod.RWS: rws,
        SelectionMethod.TOURNAMENT: tournament,
        SelectionMethod.RANK: rank
    }

    if selection not in choices:
        parent1 = random.choice(elites)
        parent2 = random.choice(elites)

    else:
        parent1 = choices[selection](population, population_fitness)
        parent2 = choices[selection](population, population_fitness)
    return parent1, parent2


def calculate_selection_pressure_fitness_variance(population_fitness):
    """
    Calculate the selection pressure (exploitation factor) using fitness variance.
    """
    population_fitness_copy = np.array(population_fitness)
    return 1 - np.var(population_fitness_copy) / np.mean(population_fitness_copy)


def calculate_selection_pressure_top_average_selection(population_fitness: [float]):
    """
    Calculate the selection pressure (exploitation factor) using top average selection.
    """
    population_fitness_copy = np.array(population_fitness)
    population_fitness_copy.sort()
    top_individuals = population_fitness_copy[:int(len(population_fitness_copy) * 0.1)]
    return np.mean(top_individuals) / np.mean(population_fitness_copy)
