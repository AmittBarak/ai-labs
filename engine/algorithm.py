import dataclasses
import random
import time
import typing
from enum import Enum

import numpy as np
from tqdm import tqdm
import coloredlogs
import logging

from engine.selection import rank, rws, sus, tournament
from engine.utils import calculate_selection_pressure_top_average_selection, calculate_selection_pressure_fitness_variance, aging

class SelectionMethod(Enum):
    SUS = 1
    RWS = 2
    TOURNAMENT = 3
    RANK = 4
    NO_SELECTION = 0

# Set up logging
coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    level_styles={
        'debug': {'color': 'green'},
        'info': {'color': 'blue'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red'},
        'critical': {'color': 'red', 'bold': True}
    },
    field_styles={
        'asctime': {'color': 'cyan'},
        'levelname': {'color': 'magenta', 'bold': True}
    }
)


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
    stop_condition_function: typing.Callable[[any, list], bool] = None
    adjust_parameters: typing.Callable = None  # Function to adjust parameters

    # Attributes for dynamic adjustment logic
    fitness_variance_threshold: float = 10.0
    top_avg_selection_threshold: float = 1.05
    adjustment_factor: float = 0.05  # 5% adjustment factor


def run_genetic_algorithm(settings: GeneticSettings):
    all_generations = []
    all_fitness_scores = []
    start_cpu_time = time.process_time()

    # Initialize the ages of the population
    ages = [random.random() for _ in range(settings.population_size)]

    # Generate the initial population
    population = [settings.individual_generator(settings.genes_count) for _ in range(settings.population_size)]

    # Check population sizes
    for individual in population:
        if len(individual) != settings.genes_count:
            logging.error(f"Error: Individual size mismatch. Expected {settings.genes_count}, got {len(individual)}")
            return

    # Run the genetic algorithm
    for generation in tqdm(range(settings.max_generations), desc="Genetic Algorithm Progress"):
        if settings.verbose:
            logging.info("\n" + "="*20)
            logging.info(f"Generation {generation}")
            logging.info("="*20)
        generation_cpu_start = time.process_time()

        # Calculate the fitness of the population
        population_fitness = [settings.fitness_calculator(individual) for individual in population]
        population_by_fitness: dict = {
            fitness: individual for individual, fitness in zip(population, population_fitness)
        }

        # Store the fitness scores and generation number
        all_fitness_scores.append(population_fitness)
        all_generations.append(generation)

        # Print the average fitness and standard deviation of the population
        avg = np.mean(population_fitness)
        dev = np.std(population_fitness)
        fitness_variance = calculate_selection_pressure_fitness_variance(population_fitness)
        top_avg_selection = calculate_selection_pressure_top_average_selection(population_fitness)
        if settings.verbose:
            logging.info(f"Standard Deviation: {dev:.6f}")
            logging.info(f"Average Fitness = {int(round(avg))}")
            logging.info(f"Selection Pressure Fitness Variance = {fitness_variance:.6f}")
            logging.info(f"Selection Pressure Top Average Selection = {top_avg_selection:.6f}")
            logging.info(f"Max fitness = {max(population_fitness)}")
            logging.info("-"*20)

        if settings.adjust_parameters:
            settings.adjust_parameters(settings, fitness_variance, top_avg_selection, generation)

        if settings.stop_condition_function is not None:
            best_individual = population_by_fitness[max(population_fitness)]
            if settings.stop_condition_function(best_individual, population_fitness):
                break

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
            logging.info(f"CPU Time for Generation {generation}: {generation_cpu_ticks:.2f} seconds")
            logging.info(f"Total Elapsed CPU Time: {generation_cpu_elapsed:.2f} seconds")
            logging.info("="*20 + "\n")

    cpu_convergence = time.process_time() - start_cpu_time
    if settings.verbose:
        logging.info(f"Total CPU Time to Convergence: {cpu_convergence:.2f} seconds")

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
