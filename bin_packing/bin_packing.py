import random
import time
import numpy as np
from engine.selection import crowding_density, nieching_partition, species_speciation

class GeneticAlgorithmBinPacking:
    def __init__(self, items, bin_capacity, population_size=100, generations=1000, mutation_rate=0.01, adaptive=False, use_aging=False):
        self.items = items
        self.bin_capacity = bin_capacity
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.adaptive = adaptive
        self.use_aging = use_aging
        self.population = self.initialize_population()
        self.recent_average_bins = None
        self.ages = [random.random() for _ in range(population_size)]
        self.best_fitness = float('inf')
        self.best_generation = 0

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = list(np.random.permutation(self.items))
            population.append(individual)
        return population

    def fitness(self, individual):
        bins = []
        current_weight = 0
        for item in individual:
            if current_weight + item <= self.bin_capacity:
                current_weight += item
            else:
                bins.append(current_weight)
                current_weight = item
        if current_weight > 0:
            bins.append(current_weight)
        return len(bins)

    def selection(self):
        if self.adaptive:
            fitness_values = [self.adaptive_fitness(ind) for ind in self.population]
        else:
            fitness_values = [self.fitness(ind) for ind in self.population]
        if self.use_aging:
            aged_fitnesses = self.aging(fitness_values, self.ages)
            sorted_population = sorted(zip(self.population, aged_fitnesses), key=lambda x: x[1])
        else:
            sorted_population = sorted(zip(self.population, fitness_values), key=lambda x: x[1])
        return [ind for ind, fit in sorted_population[:self.population_size // 2]]

    def adaptive_fitness(self, individual):
        bins = []
        current_weight = 0
        for item in individual:
            if current_weight + item <= self.bin_capacity:
                current_weight += item
            else:
                bins.append(current_weight)
                current_weight = item
        if current_weight > 0:
            bins.append(current_weight)
        num_bins = len(bins)
        fitness_score = num_bins * (
            0.9 if num_bins < self.recent_average_bins else 1.1 if num_bins > self.recent_average_bins else 1.0)
        return int(round(fitness_score))

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def aging(self, fitnesses, ages):
        young_scale, adult_scale, old_scale = 0.5, 1.5, 0.5
        thresholds = [0.25, 0.75]
        scales = [young_scale, adult_scale, old_scale]
        return [fitness * scales[sum(age > threshold for threshold in thresholds)]
                for age, fitness in zip(ages, fitnesses)]

    def run(self):
        start_time = time.process_time()
        for generation in range(self.generations):
            if self.adaptive:
                fitness_values = [self.fitness(ind) for ind in self.population]
                self.recent_average_bins = int(round(np.mean(fitness_values)))
            selected = self.selection()
            next_population = selected.copy()
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            self.population = next_population
            self.ages = [age + 1 / self.population_size for age in self.ages]
            current_best_individual = min(self.population, key=lambda x: self.fitness(x) if not self.adaptive else self.adaptive_fitness(x))
            current_best_fitness = self.fitness(current_best_individual)
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_generation = generation
        end_time = time.process_time()
        best_individual = min(self.population, key=lambda x: self.fitness(x) if not self.adaptive else self.adaptive_fitness(x))
        return best_individual, self.best_fitness, self.best_generation, end_time - start_time

    def fitness_function(self, individual, bin_capacity, method):
        if method == "nieching":
            partitions = nieching_partition(individual, bin_capacity)
            return len(partitions)
        elif method == "crowding":
            bins = crowding_density(individual, bin_capacity)
            return len(bins)
        elif method == "speciation":
            bins = species_speciation(individual, bin_capacity)
            return len(bins)
        else:
            raise ValueError("Invalid method specified.")