import random
import matplotlib.pyplot as plt
import numpy as np

class SortingNetwork:
    def __init__(self, vector_length, use_bitonic=False):
        self.vector_length = vector_length
        self.network = self.initialize_bitonic_network() if use_bitonic else self.initialize_random_network()

    def BitonicCompare(self, direction, arr):
        dist = len(arr) // 2
        for i in range(dist):
            if (arr[i] > arr[i + dist]) == direction:
                arr[i], arr[i + dist] = arr[i + dist], arr[i]

    def BitonicMerge(self, direction, arr):
        if len(arr) == 1:
            return [arr[0]]
        else:
            self.BitonicCompare(direction, arr)
            first = self.BitonicMerge(direction, arr[:len(arr) // 2])
            second = self.BitonicMerge(direction, arr[len(arr) // 2:])
            return first + second

    def BitonicSort(self, direction, arr):
        if len(arr) <= 1:
            return arr
        else:
            first = self.BitonicSort(True, arr[:len(arr) // 2])
            second = self.BitonicSort(False, arr[len(arr) // 2:])
            return self.BitonicMerge(direction, first + second)

    def initialize_bitonic_network(self):
        network = []
        # Use the BitonicSort method to initialize the network
        sorted_array = self.BitonicSort(True, list(range(self.vector_length)))
        for i in range(len(sorted_array)):
            for j in range(i+1, len(sorted_array)):
                if sorted_array[i] > sorted_array[j]:
                    network.append((i, j))
        return network

    def initialize_random_network(self):
        return [(random.randint(0, self.vector_length - 1), random.randint(0, self.vector_length - 1))
                for _ in range(int(self.vector_length * np.log2(self.vector_length)))]

    def apply(self, vector):
        vec = vector[:]
        for (i, j) in self.network:
            if 0 <= i < len(vec) and 0 <= j < len(vec):
                if vec[i] > vec[j]:
                    vec[i], vec[j] = vec[j], vec[i]
        return vec

    def mutate(self, mutation_rate):
        for i in range(len(self.network)):
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    self.indirect_replacement()
                else:
                    self.network[i] = (
                    random.randint(0, self.vector_length - 1), random.randint(0, self.vector_length - 1))
        return self

    def indirect_replacement(self):
        index_to_replace = random.randint(0, len(self.network) - 1)
        new_comparator = (random.randint(0, self.vector_length - 1), random.randint(0, self.vector_length - 1))
        self.network[index_to_replace] = new_comparator


    @staticmethod
    def crossover(parent1, parent2):
        if len(parent1.network) <= 1 or len(parent2.network) <= 1:
            # Handle the case where networks are too short to crossover
            child_network = parent1.network[:] if len(parent1.network) > len(parent2.network) else parent2.network[:]
        else:
            crossover_point = random.randint(1, len(parent1.network) - 1)
            child_network = parent1.network[:crossover_point] + parent2.network[crossover_point:]
        child = SortingNetwork(parent1.vector_length)
        child.network = child_network
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
            ax.plot([step, step], [i, j], color='black')
            ax.plot(step, i, 'ko')
            ax.plot(step, j, 'ko')
        ax.set_xlabel("Steps")
        ax.set_ylabel("Indexes")
        plt.title("Sorting Network Visualization")
        plt.show()

class GeneticAlgorithm:
    def __init__(self, population_size, vector_length, use_bitonic=False, initial_mutation_rate=0.1):
        self.vector_length = vector_length
        self.mutation_rate = initial_mutation_rate
        self.population = [SortingNetwork(vector_length, use_bitonic) for _ in range(population_size)]
        self.vectors = self.initialize_vectors(population_size, vector_length)

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

            # Gradually reduce the mutation rate
            if generation % 10 == 0 and self.mutation_rate > 0.01:
                self.mutation_rate *= 0.99

        return fitness_history

    def selection(self, fitnesses, num_offspring, tournament_size):
        selected = []
        for _ in range(num_offspring):
            tournament = random.sample(list(zip(self.population, fitnesses)), k=tournament_size)
            # Further balance fitness and comparisons
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

            if best_network_fitness == len(self.vector_population):
                print(f"Best network fitness: {best_network_fitness}, Target fitness: {len(self.vector_population)}")
                break

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

            # Gradually reduce the mutation rate
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

# def main():
#     # Parameters for K=6
#     vector_length = 6
#     population_size = 100
#     num_generations = 100
#     mutation_rate = 0.1
#     num_offspring = 50
#
#     # Initial populations for K=6
#     ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=True, initial_mutation_rate=mutation_rate)
#     fitness_history = ca.evolve(num_generations, num_offspring)
#     ca.plot_fitness(fitness_history)
#
#     # Best network found for K=6
#     best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
#     print("Best Network Found for K=6:")
#     print(best_network.network)
#
#     # Compare with QuickSort for K=6
#     correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
#     print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
#     print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")
#
#     # Parameters for K=16
#     vector_length = 16
#     population_size = 200
#     num_generations = 500
#     mutation_rate = 0.05  # Increase the mutation rate slightly for larger vector lengths
#     num_offspring = 100
#
#     # Initial populations for K=16
#     ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=True, initial_mutation_rate=mutation_rate)
#     fitness_history = ca.evolve(num_generations, num_offspring)
#     ca.plot_fitness(fitness_history)
#
#     # Best network found for K=16
#     best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
#     print("Best Network Found for K=16:")
#     print(best_network.network)
#
#     # Compare with QuickSort for K=16
#     correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
#     print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
#     print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")
#
# if __name__ == "__main__":
#     main()
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

    # Best network found for K=6
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=6:")
    print(best_network.network)
    best_network.plot_network()  # Plot the network visualization

    # Compare with QuickSort for K=6
    correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
    print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
    print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")

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

    # Best network found for K=16
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=16:")
    print(best_network.network)
    best_network.plot_network()  # Plot the network visualization

    # Compare with QuickSort for K=16
    correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
    print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
    print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")

if __name__ == "__main__":
    main()



