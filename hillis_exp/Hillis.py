import random
import matplotlib.pyplot as plt
import numpy as np
import itertools

class SortingNetwork:
    def __init__(self, vector_length, use_bitonic=False):
        self.vector_length = vector_length
        self.network = self.initialize_bitonic_network() if use_bitonic else self.initialize_random_network()

    def initialize_bitonic_network(self):
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
        network = []
        seen_pairs = set()

        while len(network) < int(self.vector_length * np.log2(self.vector_length)):
            i, j = random.randint(0, self.vector_length - 1), random.randint(0, self.vector_length - 1)
            if i != j:
                pair = tuple(sorted((i, j)))  # Sort the tuple to handle (i, j) and (j, i) as the same
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    network.append(pair)

        return network

    def apply(self, vector):
        vec = vector[:]
        for (i, j) in self.network:
            if vec[i] > vec[j]:
                vec[i], vec[j] = vec[j], vec[i]
        return vec

    def mutate(self, mutation_rate, shard_memory=None):
        for i in range(len(self.network)):
            if random.random() < mutation_rate:
                new_i, new_j = random.randint(0, self.vector_length - 1), random.randint(0, self.vector_length - 1)
                while new_i == new_j:  # Ensure no self-comparisons
                    new_j = random.randint(0, self.vector_length - 1)
                if shard_memory and random.random() < 0.7:  # Increase probability for shard memory pairs
                    new_i, new_j = random.choice(shard_memory)
                self.network[i] = (new_i, new_j)
        return self

    @staticmethod
    def crossover(parent1, parent2):
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
        fitness = 0
        for vector in vectors:
            sorted_vector = sorted(vector)
            result = self.apply(vector)
            fitness += sum(a == b for a, b in zip(result, sorted_vector)) / len(vector)
        return fitness

    def count_comparisons(self):
        return len(self.network)

    def plot_network(self):
        fig, ax = plt.subplots()
        for step, (i, j) in enumerate(self.network):
            ax.plot([step, step], [i, j], color='blue')
            ax.plot(step, i, 'ko')
            ax.plot(step, j, 'ko')
        ax.set_xlabel("Steps")
        ax.set_ylabel("Indexes")
        plt.title("Sorting Network Visualization")
        plt.show()


class ShardMemory:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.memory = []

    def add(self, network):
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)
        self.memory.append(network)

    def get_pairs(self):
        pairs = list(itertools.chain(*[net.network for net in self.memory]))
        return pairs if pairs else None


class GeneticAlgorithm:
    def __init__(self, population_size, vector_length, use_bitonic=False, initial_mutation_rate=0.1):
        self.vector_length = vector_length
        self.mutation_rate = initial_mutation_rate
        self.population = [SortingNetwork(vector_length, use_bitonic) for _ in range(population_size)]
        self.vectors = self.initialize_vectors(population_size, vector_length)
        self.shard_memory = ShardMemory()

    def initialize_vectors(self, size, vector_length):
        vectors = [random.sample(range(vector_length), vector_length) for _ in range(size)]
        print(f"Created {len(vectors)} vectors of length {vector_length}")
        return vectors

    def evolve(self, num_generations=1000, num_offspring=100, tournament_size=5):
        fitness_history = []
        for generation in range(num_generations):
            fitnesses = [network.fitness(self.vectors) for network in self.population]
            best_fitness = max(fitnesses)
            average_fitness = np.mean(fitnesses)
            best_network = self.population[fitnesses.index(best_fitness)]
            fitness_history.append((best_fitness, average_fitness, best_network.count_comparisons()))
            print(f"Generation {generation}: Best Fitness = {best_fitness}, Average Fitness = {average_fitness}, Comparisons = {best_network.count_comparisons()}")

            self.shard_memory.add(best_network)

            if best_fitness == len(self.vectors):
                print(f"Best fitness: {best_fitness}, Target fitness: {len(self.vectors)}")
                break

            elite = self.population[fitnesses.index(best_fitness)]
            selected = self.selection(fitnesses, num_offspring, tournament_size)
            offspring = [
                SortingNetwork.crossover(random.choice(selected), random.choice(selected)).mutate(self.mutation_rate, self.shard_memory.get_pairs())
                for _ in range(len(self.population) - 1)]
            offspring.append(elite)
            self.population = offspring
            self.handle_convergence_problems(fitnesses, generation, best_fitness, fitness_history)

            if generation % 10 == 0 and self.mutation_rate > 0.01:
                self.mutation_rate *= 0.99

        return fitness_history

    def selection(self, fitnesses, num_offspring, tournament_size):
        selected = []
        for _ in range(num_offspring):
            tournament = random.sample(list(zip(self.population, fitnesses)), k=tournament_size)
            tournament_winner = max(tournament, key=lambda x: x[1] - (x[0].count_comparisons() / (self.vector_length * 8 * np.log2(self.vector_length))))
            selected.append(tournament_winner[0])
        return selected

    def handle_convergence_problems(self, fitnesses, generation, best_fitness, fitness_history):
        if len(set(fitnesses)) == 1:
            self.population = [SortingNetwork(self.vector_length) for _ in range(len(self.population))]

        if max(fitnesses) - min(fitnesses) < 1:
            for network in self.population:
                network.mutate(0.5)

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

    def plot_fitness(self, fitness_history):
        best_fitness = [x[0] for x in fitness_history]
        average_fitness = [x[1] for x in fitness_history]
        comparisons = [x[2] for x in fitness_history]
        plt.plot(best_fitness, label="Best Fitness")
        plt.plot(average_fitness, label="Average Fitness")
        plt.plot(comparisons, label="Comparisons")
        plt.xlabel("Generation")
        plt.ylabel("Fitness / Comparisons")
        plt.legend()
        plt.show()

    def compare_with_quicksort(self, network, vectors):
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
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return GeneticAlgorithm.quicksort(left) + middle + GeneticAlgorithm.quicksort(right)

    def cross_validate(self, k=5):
        subset_size = len(self.vectors) // k
        scores = []
        for i in range(k):
            train_vectors = self.vectors[:i*subset_size] + self.vectors[(i+1)*subset_size:]
            test_vectors = self.vectors[i*subset_size:(i+1)*subset_size]
            best_network = max(self.population, key=lambda network: network.fitness(train_vectors))
            scores.append(best_network.fitness(test_vectors))
        return np.mean(scores), np.std(scores)

class CoevolutionaryAlgorithm(GeneticAlgorithm):
    def __init__(self, population_size, vector_length, use_bitonic=False, initial_mutation_rate=0.1):
        super().__init__(population_size, vector_length, use_bitonic, initial_mutation_rate)
        self.vector_population = [random.sample(range(vector_length), vector_length) for _ in range(population_size)]

    def evolve(self, num_generations=1000, num_offspring=100, tournament_size=5):
        fitness_history = []
        for generation in range(num_generations):
            network_fitnesses = [network.fitness(self.vector_population) for network in self.population]
            vector_fitnesses = self.evaluate_vectors(self.vector_population)

            best_network_fitness = max(network_fitnesses)
            average_network_fitness = np.mean(network_fitnesses)
            best_network = self.population[network_fitnesses.index(best_network_fitness)]
            fitness_history.append((best_network_fitness, average_network_fitness, best_network.count_comparisons()))
            print(f"Generation {generation}: Best Network Fitness = {best_network_fitness}, Average Network Fitness = {average_network_fitness}, Comparisons = {best_network.count_comparisons()}")

            self.shard_memory.add(best_network)

            if best_network_fitness == len(self.vector_population):
                print(f"Best network fitness: {best_network_fitness}, Target fitness: {len(self.vector_population)}")
                break

            elite_network = self.population[network_fitnesses.index(best_network_fitness)]
            selected_networks = self.selection(network_fitnesses, num_offspring, tournament_size)
            offspring_networks = [SortingNetwork.crossover(random.choice(selected_networks), random.choice(selected_networks)).mutate(self.mutation_rate, self.shard_memory.get_pairs()) for _ in range(len(self.population) - 1)]
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
        selected = []
        for _ in range(num_offspring):
            tournament = random.sample(list(zip(self.vector_population, fitnesses)), k=tournament_size)
            tournament_winner = min(tournament, key=lambda x: x[1])
            selected.append(tournament_winner[0])
        return selected

    @staticmethod
    def crossover_vectors(parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]


def main():

    # Parameters for K=6
    vector_length = 6
    population_size = 100
    num_generations = 100
    mutation_rate = 0.1
    num_offspring = 50

    # Initial populations for K=6
    ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=True, initial_mutation_rate=mutation_rate)
    fitness_history = ca.evolve(num_generations, num_offspring)
    ca.plot_fitness(fitness_history)

    # Cross-validation for K=6
    mean_score, std_dev = ca.cross_validate(k=5)
    print(f"Cross-Validation Mean Score for K=6: {mean_score}")
    print(f"Cross-Validation Standard Deviation for K=6: {std_dev}")

    # Best network found for K=6
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=6:")
    print(best_network.network)
    best_network.plot_network()  # Plot the network visualization

    # Compare with QuickSort for K=6
    correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
    print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
    print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")

    # Repeat the above steps for other values of K as needed
    # Parameters for K=10 and K=16 can be similarly processed

    # Parameters for K=10
    vector_length = 10
    population_size = 150
    num_generations = 500
    mutation_rate = 0.1
    num_offspring = 100

    # Initial populations for K=10
    ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=True, initial_mutation_rate=mutation_rate)
    fitness_history = ca.evolve(num_generations, num_offspring)
    ca.plot_fitness(fitness_history)

    # Cross-validation for K=10
    mean_score, std_dev = ca.cross_validate(k=5)
    print(f"Cross-Validation Mean Score for K=10: {mean_score}")
    print(f"Cross-Validation Standard Deviation for K=10: {std_dev}")

    # Best network found for K=10
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=10:")
    print(best_network.network)
    best_network.plot_network()  # Plot the network visualization

    # Compare with QuickSort for K=10
    correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
    print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
    print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")

    # Similarly for K=16
    # Parameters for K=16
    vector_length = 16
    population_size = 200
    num_generations = 500
    mutation_rate = 0.05  # Increase the mutation rate slightly for larger vector lengths
    num_offspring = 100

    # Initial populations for K=16
    ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=True, initial_mutation_rate=mutation_rate)
    fitness_history = ca.evolve(num_generations, num_offspring)
    ca.plot_fitness(fitness_history)

    # Cross-validation for K=16
    mean_score, std_dev = ca.cross_validate(k=5)
    print(f"Cross-Validation Mean Score for K=16: {mean_score}")
    print(f"Cross-Validation Standard Deviation for K=16: {std_dev}")

    # Best network found for K=16
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=16")
    print(best_network.network)
    best_network.plot_network()  # Plot the network visualization

    # Compare with QuickSort for K=16
    correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
    print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
    print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")

    # Initial populations for K=16
    ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=False, initial_mutation_rate=mutation_rate)
    fitness_history = ca.evolve(num_generations, num_offspring)
    ca.plot_fitness(fitness_history)

    # Cross-validation for K=16
    mean_score, std_dev = ca.cross_validate(k=5)
    print(f"Cross-Validation Mean Score for K=16: {mean_score}")
    print(f"Cross-Validation Standard Deviation for K=16: {std_dev}")

    # Best network found for K=16
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print(("Bitonic False"))
    print("Best Network Found for K=16")
    print(best_network.network)
    best_network.plot_network()  # Plot the network visualization

    # Compare with QuickSort for K=16
    correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
    print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
    print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")

if __name__ == "__main__":
    main()
