import abc
import dataclasses
import random
import time
from enum import Enum

import numpy as np

from engine.selection import sus, rws_linear_scaling, tournament, rank
from engine.utils import aging


class SelectionMethod(Enum):
    SUS = 1
    RWS = 2
    TOURNAMENT = 3
    RANK = 4


@dataclasses.dataclass
class GeneticConfig:
    """Configuration for the genetic algorithm."""
    use_aging: bool
    genes_count: int
    population_size: int
    max_generations: int
    mutation_rate: float
    selection: SelectionMethod
    elite_size: float = 0.1


class GeneticEngine(abc.ABC):
    """Genetic algorithm engine."""

    def __init__(self, config: GeneticConfig):
        self.population_size = config.population_size
        self.num_genes = config.genes_count
        self.max_generations = config.max_generations
        self.mutation_rate = config.mutation_rate
        self.selection = config.selection
        self.use_aging = config.use_aging
        self.elite_size = config.elite_size

    @abc.abstractmethod
    def calculate_fitness(self, individual):
        pass

    @abc.abstractmethod
    def individual_generator(self):
        pass

    @abc.abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abc.abstractmethod
    def mutate(self, individual):
        pass

    def run_genetic_algorithm(self):
        all_generations = []
        all_fitness_scores = []
        start_cpu_time = time.process_time()

        # Initialize the ages of the population
        ages = [random.random() for _ in range(self.population_size)]

        # Generate the initial
        population = [self.individual_generator() for _ in range(self.population_size)]

        # Run the genetic algorithm
        for generation in range(self.max_generations):
            print(f"Generation {generation}")
            generation_cpu_start = time.process_time()

            # Calculate the fitness of the population
            population_fitness = [self.calculate_fitness(individual) for individual in population]

            # Store the fitness scores and generation number
            all_fitness_scores.append(population_fitness)
            all_generations.append(generation)

            # Print the average fitness and standard deviation of the population
            avg = np.mean(population_fitness)
            dev = np.std(population_fitness)
            print(f"Generation {generation}")
            print(f"Best Fitness = {max(population_fitness)}")
            print(f"Worst Fitness = {min(population_fitness)}")
            print(f"Standard Deviation = {dev}")
            print(f"Average Fitness = {avg}")

            # Check for convergence
            elite_size = int(self.population_size * self.elite_size)
            elite_indices = sorted(
                range(self.population_size), key=lambda i: population_fitness[i], reverse=True
            )[:elite_size]

            # Store the elite individuals
            offspring = []
            elites = [population[i] for i in elite_indices]

            if self.use_aging:
                population_fitness = aging(population_fitness, ages)

            while len(offspring) < self.population_size - elite_size:
                parent1, parent2 = self.get_parents(population, population_fitness)

                print("Start Crossover")
                child = self.crossover(parent1, parent2)

                print("Start Mutation")
                child = self.mutate(child)
                offspring.append(child)

            print("New population")
            population = elites + offspring

            # Increment the ages of the population
            generation_cpu_end = time.process_time()
            generation_cpu_ticks = generation_cpu_end - generation_cpu_start
            generation_cpu_elapsed = generation_cpu_end - start_cpu_time

            print(f"cpu Generation {generation}: Ticks Clock cpu = {generation_cpu_ticks} seconds, "
                  f"Total Elapsed = {generation_cpu_elapsed:.2f} seconds")

        cpu_convergence = time.process_time() - start_cpu_time
        print(f"cpu Time to convergence: {cpu_convergence:.2f} seconds")

        best_individual = max(population, key=lambda individual: self.calculate_fitness(individual))
        best_fitness = self.calculate_fitness(best_individual)
        return best_individual, best_fitness, all_fitness_scores, all_generations

    def get_parents(self, population, population_fitness):
        choices: dict[SelectionMethod, callable] = {
            SelectionMethod.SUS: sus,
            SelectionMethod.RWS: rws_linear_scaling,
            SelectionMethod.TOURNAMENT: tournament,
            SelectionMethod.RANK: rank
        }

        if self.selection not in choices:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
        else:
            parent1 = choices[self.selection](population, population_fitness)
            parent2 = choices[self.selection](population, population_fitness)
        return parent1, parent2
