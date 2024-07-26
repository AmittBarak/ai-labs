import random
import matplotlib.pyplot as plt
import numpy as np
import itertools
from colorama import Fore, Style
from copy import deepcopy


class SortingNetwork:
    """
    A class representing a sorting network, which can be initialized using either a bitonic sorting network or a random network.

    Attributes:
        vector_length (int): The length of the vector to be sorted.
        network (list): The list of comparator pairs that define the sorting network.

    Methods:
        initialize_bitonic_network():
            Initializes a bitonic sorting network.

        initialize_random_network():
            Initializes a random sorting network.

        apply(vector):
            Applies the sorting network to the given vector.

        mutate(mutation_rate):
            Mutates the sorting network with a given mutation rate.

        crossover(parent1, parent2):
            Creates a new sorting network by crossing over two parent networks.

        fitness(vectors):
            Calculates the fitness of the sorting network based on how well it sorts a list of vectors.

        count_comparisons():
            Returns the number of comparisons in the sorting network.

        plot_network():
            Plots the sorting network for visualization.
    """

    def __init__(self, vector_length, use_bitonic=False):
        """
        Initializes a SortingNetwork instance.

        Args:
            vector_length (int): The length of the vector to be sorted.
            use_bitonic (bool): Whether to use a bitonic sorting network. Defaults to False.
        """
        self.vector_length = vector_length
        self.network = self.initialize_bitonic_network() if use_bitonic else self.initialize_random_network()

    def initialize_bitonic_network(self):
        """
        Initializes a bitonic sorting network.

        Returns:
            list: A list of comparator pairs defining the bitonic sorting network.
        """
        network = []
        self.temp = list(range(self.vector_length))

        def bitonic_compare(direction, low, cnt):
            dist = cnt // 2
            for i in range(low, low + dist):
                if direction == (self.temp[i] > self.temp[i + dist]):
                    self.temp[i], self.temp[i + dist] = self.temp[i + dist], self.temp[i]
                    network.append((i, i + dist))

        def bitonic_merge(direction, low, cnt):
            if cnt > 1:
                bitonic_compare(direction, low, cnt)
                k = cnt // 2
                bitonic_merge(direction, low, k)
                bitonic_merge(direction, low + k, k)

        def bitonic_sort(direction, low, cnt):
            if cnt > 1:
                k = cnt // 2
                bitonic_sort(True, low, k)
                bitonic_sort(False, low + k, k)
                bitonic_merge(direction, low, cnt)

        bitonic_sort(True, 0, self.vector_length)
        return network

    def initialize_random_network(self):
        """
        Initializes a random sorting network with a maximum number of comparators for K=16.
        """
        max_comparators = 61 if self.vector_length == 16 else int(self.vector_length * np.log2(self.vector_length))
        network = []
        seen_pairs = set()

        while len(network) < max_comparators:
            i, j = random.randint(0, self.vector_length - 1), random.randint(0, self.vector_length - 1)
            if i != j:
                pair = tuple(sorted((i, j)))  # Sort the tuple to handle (i, j) and (j, i) as the same
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    network.append(pair)

        return network

    def apply(self, vector):
        """
        Applies the sorting network to the given vector.

        Args:
            vector (list): The vector to be sorted.

        Returns:
            list: The sorted vector.
        """
        vec = vector[:]
        for (i, j) in self.network:
            if vec[i] > vec[j]:
                vec[i], vec[j] = vec[j], vec[i]
        return vec

    def mutate(self, mutation_rate):
        """
        Mutates the sorting network with a given mutation rate, ensuring the number of comparators does not exceed the limit.
        """
        max_comparators = 61 if self.vector_length == 16 else len(self.network)
        for i in range(len(self.network)):
            if random.random() < mutation_rate:
                new_i, new_j = random.randint(0, self.vector_length - 1), random.randint(0, self.vector_length - 1)
                while new_i == new_j:  # Ensure no self-comparisons
                    new_j = random.randint(0, self.vector_length - 1)
                new_pair = tuple(sorted((new_i, new_j)))
                if new_pair not in self.network:
                    self.network[i] = new_pair

        # Truncate network if it exceeds the maximum comparators
        self.network = self.network[:max_comparators]

        return self

    @staticmethod
    def crossover(parent1, parent2):
        """
        Creates a new sorting network by crossing over two parent networks.

        Args:
            parent1 (SortingNetwork): The first parent sorting network.
            parent2 (SortingNetwork): The second parent sorting network.

        Returns:
            SortingNetwork: The child sorting network created from the crossover.
        """
        crossover_point = random.randint(1, len(parent1.network) - 1)
        child_network = parent1.network[:crossover_point] + parent2.network[crossover_point:]
        child = SortingNetwork(parent1.vector_length)
        # Validate and correct self-comparisons
        corrected_network = []
        for (i, j) in child_network:
            if i == j:  # If self-comparison, pick a new index
                j = random.randint(0, parent1.vector_length - 1)
                while i == j:
                    j = random.randint(0, parent1.vector_length - 1)
            corrected_network.append((i, j))
        child.network = corrected_network
        return child

    def fitness(self, vectors):
        base_fitness = 0
        for vector in vectors:
            sorted_vector = sorted(vector)
            result = self.apply(vector)
            base_fitness += sum(a == b for a, b in zip(result, sorted_vector)) / len(vector)

        # Evaluate the complexity of the network
        complexity = evaluate_complexity(self)
        complexity_penalty = complexity / self.vector_length  # Normalize complexity penalty

        # Apply penalty only if the base fitness is very high
        if base_fitness > 0.95 * len(vectors):
            weighted_fitness = base_fitness - (0.05 * complexity_penalty)
        else:
            weighted_fitness = base_fitness

        return weighted_fitness

    def count_comparisons(self):
        """
        Returns the number of comparisons in the sorting network.

        Returns:
            int: The number of comparisons in the network.
        """
        return len(self.network)

    def plot_network(self, k, use_bitonic):
        """
        Plots the sorting network for visualization.

        Args:
            k (int): The value of K for which the network is plotted.
            use_bitonic (bool): Whether the network is bitonic or not.
        """
        fig, ax = plt.subplots()
        for step, (i, j) in enumerate(self.network):
            ax.plot([step, step], [i, j], color='blue')
            ax.plot(step, i, 'ko')
            ax.plot(step, j, 'ko')
        ax.set_xlabel("Steps")
        ax.set_ylabel("Indexes")
        title = f"Sorting Network for K={k} {'(Bitonic)' if use_bitonic else '(Random)'}"
        plt.title(title)
        plt.show()

class GeneticAlgorithm:
    """
    A class representing a genetic algorithm for evolving sorting networks.

    Attributes:
        vector_length (int): The length of the vectors to be sorted.
        mutation_rate (float): The initial mutation rate for the sorting networks.
        population (list): The list of sorting networks in the population.
        vectors (list): The list of vectors to be used for evaluating fitness.

    Methods:
        initialize_vectors(size, vector_length):
            Initializes a list of vectors with random permutations.

        evolve(num_generations, num_offspring, tournament_size):
            Evolves the population of sorting networks over a number of generations.

        selection(fitnesses, num_offspring, tournament_size):
            Selects a number of sorting networks from the population based on their fitness.

        handle_convergence_problems(fitnesses, generation, best_fitness, fitness_history):
            Handles potential convergence problems in the genetic algorithm.

        plot_fitness(fitness_history):
            Plots the fitness history of the genetic algorithm.

        compare_with_quicksort(network, vectors):
            Compares the performance of the sorting network with the quicksort algorithm.

        quicksort(arr):
            Implements the quicksort algorithm for comparison purposes.

        cross_validate(k):
            Performs k-fold cross-validation on the population of sorting networks.
    """

    def __init__(self, population_size, vector_length, use_bitonic=False, initial_mutation_rate=0.1):
        """
        Initializes a GeneticAlgorithm instance.

        Args:
            population_size (int): The size of the population.
            vector_length (int): The length of the vectors to be sorted.
            use_bitonic (bool): Whether to use bitonic sorting networks. Defaults to False.
            initial_mutation_rate (float): The initial mutation rate for the sorting networks. Defaults to 0.1.
        """
        self.vector_length = vector_length
        self.mutation_rate = initial_mutation_rate
        self.population = [SortingNetwork(vector_length, use_bitonic) for _ in range(population_size)]
        self.vectors = self.initialize_vectors(population_size, vector_length)

    def initialize_vectors(self, size, vector_length):
        """
        Initializes a list of vectors with random permutations.

        Args:
            size (int): The number of vectors to initialize.
            vector_length (int): The length of each vector.

        Returns:
            list: A list of vectors with random permutations.
        """
        vectors = [random.sample(range(vector_length), vector_length) for _ in range(size)]
        print(f"Created {len(vectors)} vectors of length {vector_length}")
        return vectors

    def evolve(self, num_generations=1000, num_offspring=100, tournament_size=5):
        """
        Evolves the population of sorting networks over a number of generations.

        Args:
            num_generations (int): The number of generations to evolve. Defaults to 1000.
            num_offspring (int): The number of offspring to generate in each generation. Defaults to 100.
            tournament_size (int): The size of the tournament for selection. Defaults to 5.

        Returns:
            list: A list of tuples representing the fitness history of the genetic algorithm.
        """
        fitness_history = []
        for generation in range(num_generations):
            fitnesses = [network.fitness(self.vectors) for network in self.population]
            best_fitness = max(fitnesses)
            average_fitness = np.mean(fitnesses)
            best_network = self.population[fitnesses.index(best_fitness)]
            fitness_history.append((best_fitness, average_fitness, best_network.count_comparisons()))
            print(f"Generation {generation}: Best Fitness = {best_fitness}, Average Fitness = {average_fitness}, Comparisons = {best_network.count_comparisons()}")

            if best_fitness == len(self.vectors):
                print(f"Best fitness: {best_fitness}, Target fitness: {len(self.vectors)}")
                break


            elite = self.population[fitnesses.index(best_fitness)]
            selected = self.selection(fitnesses, num_offspring, tournament_size)
            offspring = [
                SortingNetwork.crossover(random.choice(selected), random.choice(selected)).mutate(self.mutation_rate)
                for _ in range(len(self.population) - 1)]
            offspring.append(elite)
            self.population = offspring
            self.handle_convergence_problems(fitnesses, generation, best_fitness, fitness_history)

            if generation % 10 == 0 and self.mutation_rate > 0.01:
                self.mutation_rate *= 0.99

        return fitness_history

    def selection(self, fitnesses, num_offspring, tournament_size):
        """
        Selects a number of sorting networks from the population based on their fitness.

        Args:
            fitnesses (list): The list of fitness scores for the population.
            num_offspring (int): The number of offspring to select.
            tournament_size (int): The size of the tournament for selection.

        Returns:
            list: The list of selected sorting networks.
        """
        selected = []
        for _ in range(num_offspring):
            tournament = random.sample(list(zip(self.population, fitnesses)), k=tournament_size)
            tournament_winner = max(tournament, key=lambda x: x[1] - (x[0].count_comparisons() /
                                                                      (self.vector_length * 8 * np.log2(self.vector_length))))
            selected.append(tournament_winner[0])
        return selected

    def handle_convergence_problems(self, fitnesses, generation, best_fitness, fitness_history):
        """
        Handles potential convergence problems in the genetic algorithm.

        Args:
            fitnesses (list): The list of fitness scores for the population.
            generation (int): The current generation number.
            best_fitness (float): The best fitness score in the current generation.
            fitness_history (list): The history of fitness scores.
        """
        if len(set(fitnesses)) == 1:
            self.population = [SortingNetwork(self.vector_length) for _ in range(len(self.population))]

        if max(fitnesses) - min(fitnesses) < 1:
            for network in self.population:
                network.mutate(0.5)

        std_dev = np.std(fitnesses)
        if std_dev < 0.01:  # Low standard deviation indicates lack of diversity
            self.mutation_rate += 0.01  # Increase mutation rate
        elif std_dev > 0.1:
            self.mutation_rate = max(0.01, self.mutation_rate - 0.01)  # Decrease mutation rate to stabilize the search

        # Regenerate part of the population if there is no improvement
        if generation > 1 and best_fitness <= fitness_history[-1][0]:
            num_to_reinitialize = len(self.population) // 5
            for i in range(num_to_reinitialize):
                self.population[i] = SortingNetwork(self.vector_length)
                self.population[i].mutate(self.mutation_rate)

        if generation > 0 and best_fitness <= fitness_history[-1][0]:
            num_to_reinitialize = len(self.population) // 10
            for i in range(num_to_reinitialize):
                self.population[i] = SortingNetwork(self.vector_length)

        for i in range(len(self.population)):
            if fitnesses[i] == max(fitnesses):
                self.population[i].mutate(0.5)

        if np.std(fitnesses) < 0.01:
            num_to_reinitialize = len(self.population) // 5
            for i in range(num_to_reinitialize):
                self.population[i] = SortingNetwork(self.vector_length)

    def plot_fitness(self, fitness_history, k, use_bitonic):
        """
        Plots the fitness history of the genetic algorithm.

        Args:
            fitness_history (list): The history of fitness scores.
            k (int): The value of K for which the fitness is plotted.
            use_bitonic (bool): Whether the network is bitonic or not.
        """
        best_fitness = [x[0] for x in fitness_history]
        average_fitness = [x[1] for x in fitness_history]
        comparisons = [x[2] for x in fitness_history]

        plt.figure()
        plt.plot(best_fitness, label="Best Fitness")
        plt.plot(average_fitness, label="Average Fitness")
        plt.plot(comparisons, label="Comparisons")
        plt.xlabel("Generation")
        plt.ylabel("Fitness / Comparisons")
        title = f"Fitness History for K={k} {'(Bitonic)' if use_bitonic else '(Random)'}"
        plt.title(title)
        plt.legend()
        plt.show()

    def compare_with_quicksort(self, network, vectors):
        """
        Compares the performance of the sorting network with the quicksort algorithm.

        Args:
            network (SortingNetwork): The sorting network to compare.
            vectors (list): The list of vectors to sort.

        Returns:
            tuple: The number of correctly sorted vectors by the network and by quicksort.
        """
        correct = 0
        quicksort_correct = 0
        for vector in vectors:
            sorted_vector = sorted(vector)
            if network.apply(vector) == sorted_vector:
                correct += 1
            if self.quicksort(vector) == sorted_vector:
                quicksort_correct += 1
        return correct, quicksort_correct

    @staticmethod
    def quicksort(arr):
        """
        Implements the quicksort algorithm for comparison purposes.

        Args:
            arr (list): The list to sort.

        Returns:
            list: The sorted list.
        """
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return GeneticAlgorithm.quicksort(left) + middle + GeneticAlgorithm.quicksort(right)

    def cross_validate(self, k=5):
        """
        Performs k-fold cross-validation on the population of sorting networks.

        Args:
            k (int): The number of folds for cross-validation. Defaults to 5.

        Returns:
            tuple: The mean and standard deviation of the cross-validation scores.
        """
        subset_size = len(self.vectors) // k
        scores = []
        for i in range(k):
            train_vectors = self.vectors[:i*subset_size] + self.vectors[(i+1)*subset_size:]
            test_vectors = self.vectors[i*subset_size:(i+1)*subset_size]
            best_network = max(self.population, key=lambda network: network.fitness(train_vectors))
            scores.append(best_network.fitness(test_vectors))
        return np.mean(scores), np.std(scores)

class CoevolutionaryAlgorithm(GeneticAlgorithm):
    """
    A class representing a coevolutionary algorithm for evolving sorting networks alongside vectors.

    Attributes:
        vector_population (list): The population of vectors to coevolve with the sorting networks.

    Methods:
        evolve(num_generations, num_offspring, tournament_size):
            Evolves the population of sorting networks and vectors over a number of generations.

        evaluate_vectors(vectors):
            Evaluates the fitness of each vector based on how well the sorting networks sort them.

        selection_vectors(fitnesses, num_offspring, tournament_size):
            Selects a number of vectors from the population based on their fitness.

        crossover_vectors(parent1, parent2):
            Creates a new vector by crossing over two parent vectors.
    """

    def __init__(self, population_size, vector_length, use_bitonic=False, initial_mutation_rate=0.1):
        """
        Initializes a CoevolutionaryAlgorithm instance.

        Args:
            population_size (int): The size of the population.
            vector_length (int): The length of the vectors to be sorted.
            use_bitonic (bool): Whether to use bitonic sorting networks. Defaults to False.
            initial_mutation_rate (float): The initial mutation rate for the sorting networks. Defaults to 0.1.
        """
        super().__init__(population_size, vector_length, use_bitonic, initial_mutation_rate)
        self.vector_population = [random.sample(range(vector_length), vector_length) for _ in range(population_size)]

    def evolve(self, num_generations=1000, num_offspring=100, tournament_size=5):
        """
        Evolves the population of sorting networks and vectors over a number of generations.

        Args:
            num_generations (int): The number of generations to evolve. Defaults to 1000.
            num_offspring (int): The number of offspring to generate in each generation. Defaults to 100.
            tournament_size (int): The size of the tournament for selection. Defaults to 5.

        Returns:
            list: A list of tuples representing the fitness history of the coevolutionary algorithm.
        """
        fitness_history = []
        required_fitness = len(self.vector_population)
        acceptable_margin = 0.05 * required_fitness

        for generation in range(num_generations):
            network_fitnesses = [network.fitness(self.vector_population) for network in self.population]
            vector_fitnesses = self.evaluate_vectors(self.vector_population)

            best_network_fitness = max(network_fitnesses)
            average_network_fitness = np.mean(network_fitnesses)
            best_network = self.population[network_fitnesses.index(best_network_fitness)]
            fitness_history.append((best_network_fitness, average_network_fitness, best_network.count_comparisons()))

            if best_network_fitness == len(self.vector_population):
                print(f"Best network fitness: {best_network_fitness}, Target fitness: {len(self.vector_population)}")
                break
            if best_network_fitness >= required_fitness - acceptable_margin:
                print(
                    f"Found the best generation {generation}: Best Network Fitness = {required_fitness}, Average Network Fitness = "
                    f"{average_network_fitness}, Comparisons = {best_network.count_comparisons()}")

                break
            print(f"Generation {generation}: Best Network Fitness = {best_network_fitness}, Average Network Fitness = {average_network_fitness}, Comparisons = {best_network.count_comparisons()}")


            elite_network = self.population[network_fitnesses.index(best_network_fitness)]
            selected_networks = self.selection(network_fitnesses, num_offspring, tournament_size)
            offspring_networks = [SortingNetwork.crossover(random.choice(selected_networks), random.choice(selected_networks)).mutate(self.mutation_rate) for _ in range(len(self.population) - 1)]
            offspring_networks.append(elite_network)
            self.population = offspring_networks

            elite_vector = self.vector_population[vector_fitnesses.index(min(vector_fitnesses))]
            selected_vectors = self.selection_vectors(vector_fitnesses, num_offspring, tournament_size)
            offspring_vectors = [self.crossover_vectors(random.choice(selected_vectors), random.choice(selected_vectors)) for _ in range(len(self.vector_population) - 1)]
            offspring_vectors.append(elite_vector)
            self.vector_population = offspring_vectors

            self.handle_convergence_problems(network_fitnesses, generation, best_network_fitness, fitness_history)

            if generation % 10 == 0 and self.mutation_rate > 0.01:
                self.mutation_rate *= 0.99

        return fitness_history

    def evaluate_vectors(self, vectors):
        """
        Evaluates the fitness of each vector based on how well the sorting networks sort them.

        Args:
            vectors (list): The list of vectors to evaluate.

        Returns:
            list: The list of fitness scores for each vector.
        """
        vector_fitnesses = []
        for vector in vectors:
            fitness = 0
            for network in self.population:
                sorted_vector = sorted(vector)
                result = network.apply(vector)
                fitness += sum(a == b for a, b in zip(result, sorted_vector)) / len(vector)
            vector_fitnesses.append(fitness)
        return vector_fitnesses

    def selection_vectors(self, fitnesses, num_offspring, tournament_size):
        """
        Selects a number of vectors from the population based on their fitness.

        Args:
            fitnesses (list): The list of fitness scores for the vector population.
            num_offspring (int): The number of offspring to select.
            tournament_size (int): The size of the tournament for selection.

        Returns:
            list: The list of selected vectors.
        """
        selected = []
        for _ in range(num_offspring):
            tournament = random.sample(list(zip(self.vector_population, fitnesses)), k=tournament_size)
            tournament_winner = min(tournament, key=lambda x: x[1])
            selected.append(tournament_winner[0])
        return selected

    @staticmethod
    def crossover_vectors(parent1, parent2):
        """
        Creates a new vector by crossing over two parent vectors.

        Args:
            parent1 (list): The first parent vector.
            parent2 (list): The second parent vector.

        Returns:
            list: The child vector created from the crossover.
        """
        crossover_point = random.randint(1, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

def evaluate_complexity(network):
    """
    Calculate the maximum depth of comparisons in a given sorting network to evaluate its complexity.
    This depth indicates the worst-case scenario in terms of the number of layers a value might pass through.

    Parameters:
    - network (SortingNetwork): An object representing the sorting network with attributes:
      - vector_length (int): Length of vectors that the network sorts.
      - network (list of tuples): List where each tuple (i, j) represents a comparison between indices i and j of a vector.

    Returns:
    - int: The maximum depth of comparisons in the network.
    """
    depths = [0] * network.vector_length
    for (i, j) in network.network:
        depths[j] = max(depths[i] + 1, depths[j])
    max_depth = max(depths)
    return max_depth
