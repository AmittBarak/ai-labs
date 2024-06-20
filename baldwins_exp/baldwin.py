import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

class EvolutionaryAlgorithm:
    def __init__(self, target_pattern, population_size=1000, generations=100, mutation_rate=0.01):
        self.target_pattern = np.array(target_pattern)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.pattern_length = len(target_pattern)
        self.population = self.initialize_population(population_size, self.pattern_length)

    def initialize_population(self, size, pattern_length):
        return np.random.choice(['0', '1', '?', '{', '}'], (size, pattern_length))

    def memetic_algorithm(self, creature, attempts=1000):
        learned_pattern = creature.copy()
        for _ in range(attempts):
            question_mask = (learned_pattern == '?')
            learned_pattern[question_mask] = np.random.choice(['0', '1', '{', '}'], np.sum(question_mask))
            mismatch_mask = (learned_pattern != self.target_pattern) & (learned_pattern != '?')
            learned_pattern[mismatch_mask] = '?'
        return learned_pattern

    def apply_memetic_algorithm(self):
        with Pool(cpu_count()) as p:
            learned_population = p.starmap(self.memetic_algorithm, [(creature,) for creature in self.population])
        self.population = np.array(learned_population)

    def fitness(self, creature):
        return np.sum(creature == self.target_pattern)

    def selection(self):
        fitness_scores = np.array([self.fitness(creature) for creature in self.population])
        probabilities = fitness_scores / np.sum(fitness_scores)
        selected_indices = np.random.choice(len(self.population), size=len(self.population), p=probabilities)
        self.population = self.population[selected_indices]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutation(self, creature):
        mutation_mask = np.random.rand(len(creature)) < self.mutation_rate
        creature[mutation_mask] = np.random.choice(['0', '1', '{', '}'], np.sum(mutation_mask))
        return creature

    def evolve_population(self):
        self.selection()
        new_population = []
        for i in range(0, len(self.population), 2):
            parent1, parent2 = self.population[i], self.population[i+1]
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutation(child1))
            new_population.append(self.mutation(child2))
        self.population = np.array(new_population)

    def calculate_metrics(self):
        correct_matches = np.mean([np.sum(creature == self.target_pattern) for creature in self.population]) / self.pattern_length
        incorrect_positions = np.mean([np.sum((creature != self.target_pattern) & (creature != '?')) for creature in self.population]) / self.pattern_length
        learned_bits = np.mean([np.sum(creature != '?') for creature in self.population]) / self.pattern_length
        return correct_matches, incorrect_positions, learned_bits

    def run_simulation(self, with_learning=True):
        correct_matches_list, incorrect_positions_list, learned_bits_list = [], [], []

        for generation in range(self.generations):
            if with_learning:
                self.apply_memetic_algorithm()
            correct_matches, incorrect_positions, learned_bits = self.calculate_metrics()
            correct_matches_list.append(correct_matches)
            incorrect_positions_list.append(incorrect_positions)
            learned_bits_list.append(learned_bits)
            self.evolve_population()

        return correct_matches_list, incorrect_positions_list, learned_bits_list
