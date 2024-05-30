import dataclasses
import random
import time
import typing
from enum import Enum

import numpy as np

from engine.selection import sus, rws_linear_scaling, tournament, rank
from sudoku.utils import aging, print_pretty_grid


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
    elite_size: float = 0.1
    mutation_generator: typing.Callable[[any], any] = None
    crossover_generator: typing.Callable[[any, any], any] = None
    fitness_calculator: typing.Callable[[any], float] = None
    individual_generator: typing.Callable[[], any] = None


def run_genetic_algorithm(settings: GeneticSettings):
    all_generations = []
    all_fitness_scores = []
    start_cpu_time = time.process_time()

    # Initialize the ages of the population
    ages = [random.random() for _ in range(settings.population_size)]

    # Generate the initial
    population = [settings.individual_generator() for _ in range(settings.population_size)]

    # Check population sizes
    for individual in population:
        if len(individual) != settings.genes_count:
            print(f"Error: Individual size mismatch. Expected {settings.genes_count}, got {len(individual)}")
            return

    # Run the genetic algorithm
    for generation in range(settings.max_generations):
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

        # Print the best individual every 10 generations
        if generation % 10 == 0:
            best_individual = max(population, key=lambda i: settings.fitness_calculator(i))
            print_pretty_grid(best_individual)

        print(f"Generation {generation}")
        print(f"Best Fitness = {max(population_fitness)}")
        print(f"Worst Fitness = {min(population_fitness)}")
        print(f"Standard Deviation = {dev}")
        print(f"Average Fitness = {avg}")

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
            parent1, parent2 = get_parents(settings.selection, population, population_fitness)
            child1, child2 = settings.crossover_generator(parent1, parent2)

            # validate that we got a permutation
            if sorted(child1) != sorted(parent1) or sorted(child2) != sorted(parent2):
                print(f"Error: Crossover did not produce a permutation.")
                return

            if len(child1) != settings.genes_count or len(child2) != settings.genes_count:
                print(f"Error: Crossover resulted in incorrect child size.")
                return

            if random.random() < settings.mutation_rate:
                child1 = settings.mutation_generator(child1)
                child2 = settings.mutation_generator(child2)

            # validate that we got a permutation
            if sorted(child1) != sorted(parent1) or sorted(child2) != sorted(parent2):
                print(f"Error: Mutate did not produce a permutation.")
                return

            if len(child1) != settings.genes_count or len(child2) != settings.genes_count:
                print(f"Error: Mutation resulted in incorrect child size.")
                return

            offspring.append(child1)
            offspring.append(child2)

        population = elites + offspring[:settings.population_size - elite_size]

        # Increment the ages of the population
        generation_cpu_end = time.process_time()
        generation_cpu_ticks = generation_cpu_end - generation_cpu_start
        generation_cpu_elapsed = generation_cpu_end - start_cpu_time

        print(f"cpu Generation {generation}: Ticks Clock cpu = {generation_cpu_ticks} seconds, "
              f"Total Elapsed = {generation_cpu_elapsed:.2f} seconds")

    cpu_convergence = time.process_time() - start_cpu_time
    print(f"cpu Time to convergence: {cpu_convergence:.2f} seconds")

    best_individual = max(population, key=lambda i: settings.fitness_calculator(i))
    best_fitness = settings.fitness_calculator(best_individual)
    return best_individual, best_fitness, all_fitness_scores, all_generations


def get_parents(selection, population, population_fitness):
    choices: dict[SelectionMethod, callable] = {
        SelectionMethod.SUS: sus,
        SelectionMethod.RWS: rws_linear_scaling,
        SelectionMethod.TOURNAMENT: tournament,
        SelectionMethod.RANK: rank
    }

    if selection not in choices:
        parent1 = random.choice(population)
        parent2 = random.choice(population)

    else:
        parent1 = choices[selection](population, population_fitness)
        parent2 = choices[selection](population, population_fitness)
    return parent1, parent2
