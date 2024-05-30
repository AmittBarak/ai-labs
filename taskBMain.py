import random
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
from typing import List, Dict, Tuple
from collections import defaultdict
from sudoku import solver as sudoku_solver


def initialize_population(items, population_size):
    try:
        population = []
        for _ in range(population_size):
            try:
                individual = list(np.random.permutation(items))
                population.append(individual)
            except (ValueError, TypeError):
                print(f"Error: Unable to generate a valid individual from the provided items.")
                continue
        return population
    except TypeError:
        print("Error: The items parameter is not iterable or population_size is not an integer.")
        return None


def fitness(individual, bin_capacity):
    """
    Fixed Fitness function, how?
    Consistency: The function evaluates the fitness of an individual using a consistent formula.
    It calculates the number of bins required to pack all items without changing its criteria or formula based on
    performance or any other feedback mechanism.

    Non-Adaptive: The function does not adjust or adapt based on the performance of individuals during the
    evolutionary process.
    It always uses the same method to determine the number of bins required, regardless of how the
    solutions evolve over time.
    """
    try:
        bins = []
        current_weight = 0

        for item in individual:
            try:
                if current_weight + item <= bin_capacity:
                    current_weight += item
                else:
                    bins.append(current_weight)
                    current_weight = item
            except TypeError:
                print(f"Error: An item in the individual ({item}) is not a valid weight.")
                return None

        if current_weight > 0:
            bins.append(current_weight)

        return len(bins)
    except TypeError:
        print("Error: The individual or bin_capacity parameter is not of the correct type.")
        return None


def adaptive_fitness(individual, bin_capacity, recent_average_bins):
    """
    Self adaptive Fitness function, why?
    This self-adapting fitness function because it dynamically adjusts based on recent average performance,
    ensuring continuous improvement and responsiveness to the population's current state.
    Its feedback mechanism rewards better solutions and penalizes worse ones, effectively guiding the algorithm toward
    optimal solutions. This adaptability maintains a balance between exploration and exploitation,
    preventing premature convergence. Additionally, it provides a holistic adjustment by incorporating overall
    population trends, leading to a more robust and efficient optimization process.
    """
    try:
        bins, current_weight = [], 0

        try:
            for item in individual:
                if current_weight + item <= bin_capacity:
                    current_weight += item
                else:
                    bins.append(current_weight)
                    current_weight = item
            if current_weight > 0:
                bins.append(current_weight)
        except TypeError:
            print(f"Error: Invalid item weight {item} in individual.")
            return None

        num_bins = len(bins)
        fitness_score = num_bins * (
            0.9 if num_bins < recent_average_bins else 1.1 if num_bins > recent_average_bins else 1.0)

        return fitness_score
    except TypeError:
        print("Error: The individual or bin_capacity parameter is not of the correct type.")
        return None


def selection(population, bin_capacity, population_size, adaptive=False, recent_average_bins=None):
    if adaptive:
        sorted_population = sorted(population, key=lambda x: adaptive_fitness(x, bin_capacity, recent_average_bins))
    else:
        sorted_population = sorted(population, key=lambda x: fitness(x, bin_capacity))
    return sorted_population[:population_size // 2]

# This is exectly like single
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate_for_bin_packing(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def genetic_algorithm_bin_packing(items, bin_capacity, population_size=100, generations=1000, mutation_rate=0.01,
                                  adaptive=False):
    start_time = time.process_time()
    try:
        population = initialize_population(items, population_size)
        if population is None:
            return None, None

        recent_average_bins = None

        for generation in range(generations):
            if adaptive:
                fitness_values = [fitness(ind, bin_capacity) for ind in population]
                recent_average_bins = int(round(np.mean(fitness_values)))

            try:
                selected = selection(population, bin_capacity, population_size, adaptive, recent_average_bins)
            except Exception as e:
                print(f"Error in selection: {e}")
                return None, None

            next_population = selected.copy()

            while len(next_population) < population_size:
                try:
                    parent1, parent2 = random.sample(selected, 2)
                    child1, child2 = crossover(parent1, parent2)
                    next_population.append(mutate_for_bin_packing(child1, mutation_rate))
                    next_population.append(mutate_for_bin_packing(child2, mutation_rate))
                except Exception as e:
                    print(f"Error in crossover or mutation: {e}")
                    continue

            population = next_population

        if adaptive:
            best_individual = min(population, key=lambda x: adaptive_fitness(x, bin_capacity, recent_average_bins))
            best_fitness = adaptive_fitness(best_individual, bin_capacity, recent_average_bins)
        else:
            best_individual = min(population, key=lambda x: fitness(x, bin_capacity))
            best_fitness = fitness(best_individual, bin_capacity)

        if best_fitness is None:
            return None, None

        end_time = time.process_time()

        return best_individual, best_fitness
    except Exception as e:
        print(f"Error in genetic_algorithm_bin_packing: {e}")
        return None, None


def aging(population, fitnesses, age_distribution):
    """
    This function takes a population's fitnesses and age distribution, and
    applies a scaling factor based on age. The scaling factors are:
    - 0.5 for ages below 0.25 (young)
    - 1.5 for ages between 0.25 and 0.75 (adult)
    - 0.5 for ages above 0.75 (old)
    """
    young_scale, adult_scale, old_scale = 0.5, 1.5, 0.5
    thresholds = [0.25, 0.75]
    scales = [young_scale, adult_scale, old_scale]

    return [fitness * scales[sum(age > threshold for threshold in thresholds)]
            for age, fitness in zip(age_distribution, fitnesses)]


def rws_linear_scaling(population, fitnesses, a=1.0, b=0.0, min_val=0.0, max_val=1.0):
    fitnesses = np.array(fitnesses)
    min_fitness = fitnesses.min()
    max_fitness = fitnesses.max()

    # Apply linear scaling to fitness values
    scaled_fitnesses = a * (fitnesses - min_fitness) / (max_fitness - min_fitness) + b
    scaled_fitnesses = np.clip(scaled_fitnesses, min_val, max_val)
    scaled_fitnesses = min_val + scaled_fitnesses * (max_val - min_val)

    # Handle the case where all fitness values are the same
    if max_fitness == min_fitness:
        scaled_fitnesses = np.full_like(scaled_fitnesses, min_val + (max_val - min_val) / 2)

    # Compute selection probabilities and select an individual
    selection_probs = scaled_fitnesses / scaled_fitnesses.sum()
    selected_index = np.random.choice(len(population), p=selection_probs)

    return population[selected_index]


def sus(population, fitnesses, a=1.0, b=0.0, min_val=0.0, max_val=1.0):
    fitnesses = np.array(fitnesses)
    min_fitness = fitnesses.min()
    max_fitness = fitnesses.max()

    # Apply linear scaling to fitness values
    if max_fitness != min_fitness:
        scaled_fitnesses = a * (fitnesses - min_fitness) / (max_fitness - min_fitness) + b
        scaled_fitnesses = min_val + scaled_fitnesses * (max_val - min_val)
    else:
        # Handle the case where all fitness values are the same
        scaled_fitnesses = np.full_like(fitnesses, min_val + (max_val - min_val) / 2)

    # Compute selection probabilities
    total_fitness = np.sum(scaled_fitnesses)
    selection_probs = scaled_fitnesses / total_fitness

    num_selections = len(population)
    start = np.random.uniform(0, 1.0 / num_selections)
    pointers = start + np.arange(num_selections) * 1.0 / num_selections
    cumsum = np.cumsum(selection_probs)

    selected_indices = []
    i = 0
    for p in pointers:
        while p > cumsum[i]:
            i += 1
        selected_indices.append(i)

    # Select individuals based on the indices
    selected_individuals = [population[idx] for idx in selected_indices]

    return selected_individuals


def rank(population, fitnesses):
    ranks = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
    rank_probs = [rank / sum(ranks) for rank in ranks]
    return population[np.random.choice(len(population), p=rank_probs)]


def tournament(population, fitnesses, k=3, p=0.7):
    # If we'll make k=1 it's a 1-way tournament which is random so we'll make it a bit less random
    # If we'll make p=1 the selection becomes Deterministic and we don't want that
    selected = random.sample(range(len(population)), k)
    best_idx = max(selected, key=lambda i: fitnesses[i])

    if random.random() < p:
        return population[best_idx]
    else:
        selected.remove(best_idx)
        return population[random.choice(selected)]


def fitness_GA(individual):
    """
    Calculates the fitness score of a string.
    This function takes a single argument which is a string. It calculates a score based on how closely the string
    matches the target string "Hello, world!".
    The score is calculated by comparing the characters in the string to the corresponding characters in the target
    string.
    If a character in the string matches the corresponding character in the target string, the score is incremented by 1.

    Returns:
        The fitness score of the string. The higher the score, the more similar the string is to the target string.
    """
    """better way to write:
    target = "Hello, world!"
    return sum(char1 == char2 for char1, char2 in zip(individual, target))
    """
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score


def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, mutation_rate, selection, use_aging):
    try:
        start_CPU_time = time.process_time()
        all_fitness_scores = []
        all_generations = []
        population = [[''.join(chr(random.randint(32, 126)) for _ in range(num_genes))] for _ in range(pop_size)]
        ages = [random.random() for _ in range(pop_size)]

        for generation in range(max_generations):
            generation_CPU_start = time.process_time()
            try:
                fitnesses = [fitness_func(individual[0]) for individual in population]
            except Exception as e:
                print(f"Error in fitness function: {e}")
                return None, None, None, None

            all_fitness_scores.append(fitnesses)
            all_generations.append(generation)

            avg = np.mean(fitnesses)
            dev = np.std(fitnesses)
            print(f"Generation {generation}: Average Fitness = {int(round(avg))}, Standard Deviation = {int(round(dev))}")

            elite_size = int(pop_size * 0.1)
            elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
            elites = [population[i] for i in elite_indices]

            offspring = []
            if use_aging:
                aging_fitnesses = aging(population, fitnesses, ages)
                while len(offspring) < pop_size - elite_size:
                    if selection == "2":
                        parent1, parent2 = rws_linear_scaling(population, aging_fitnesses)[0], rws_linear_scaling(population, aging_fitnesses)[0]
                    elif selection == "1":
                        parent1, parent2 = sus(population, aging_fitnesses)[0], sus(population, aging_fitnesses)[0]
                    elif selection == "3":
                        parent1, parent2 = tournament(population, aging_fitnesses), tournament(population, aging_fitnesses)
                    elif selection == "4":
                        parent1, parent2 = rank(population, aging_fitnesses), rank(population, aging_fitnesses)
                    else:
                        parent1, parent2 = random.choice(elites)[0], random.choice(elites)[0]
                    child = ''.join([parent1[i] if random.random() < 0.5 else parent2[i] for i in range(min(len(parent1), len(parent2)))])
                    child = mutate(child, mutation_rate)
                    offspring.append([child])
            else:
                while len(offspring) < pop_size - elite_size:
                    if selection == "2":
                        parent1, parent2 = rws_linear_scaling(population, fitnesses)[0], rws_linear_scaling(population, fitnesses)[0]
                    elif selection == "1":
                        parent1, parent2 = sus(population, fitnesses)[0], sus(population, fitnesses)[0]
                    elif selection == "3":
                        parent1, parent2 = tournament(population, fitnesses), tournament(population, fitnesses)
                    elif selection == "4":
                        parent1, parent2 = rank(population, fitnesses), rank(population, fitnesses)
                    else:
                        parent1, parent2 = random.choice(elites)[0], random.choice(elites)[0]
                    child = ''.join([parent1[i] if random.random() < 0.5 else parent2[i] for i in range(min(len(parent1), len(parent2)))])
                    child = mutate(child, mutation_rate)
                    offspring.append([child])
            population = elites + offspring

            generation_CPU_end = time.process_time()
            generation_CPU_ticks = generation_CPU_end - generation_CPU_start
            generation_CPU_elapsed = generation_CPU_end - start_CPU_time

            print(f"CPU Generation {generation}: Ticks Clock CPU = {generation_CPU_ticks} seconds, "
                  f"Total Elapsed = {generation_CPU_elapsed:.2f} seconds")

        CPU_convergence = time.process_time() - start_CPU_time
        print(f"CPU Time to convergence: {CPU_convergence:.2f} seconds")

        best_individual = max(population, key=lambda individual: fitness_func(individual[0]))
        best_fitness = fitness_func(best_individual[0])

        return best_individual[0], best_fitness, all_fitness_scores, all_generations
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None


def tournament_for_bin_packing(population, fitnesses, k=3):
    if len(population) != len(fitnesses):
        raise IndexError("Population and fitnesses lists must have the same length.")

    # Check if the population and fitnesses lists are not empty
    if not population or not fitnesses:
        raise ValueError("Population and fitnesses lists cannot be empty.")

    # Check if the tournament size is valid
    if k < 2:
        raise ValueError("Tournament size (k) must be at least 2.")
    try:
        selected = []
        # Perform tournament selection
        for _ in range(len(population) // 2):
            tournament = random.sample(list(zip(population, fitnesses)), k)
            # Find the individual with the minimum fitness (best individual) in the tournament
            best_individual, _ = min(tournament, key=lambda x: x[1])
            # Add the best individual to the selected list
            selected.append(best_individual)
        return selected
    except Exception as e:
        print(f"An error occurred: {e}")


def rank_bin_packing(population, fitnesses):
    # Check if the population and fitnesses lists have the same length
    if len(population) != len(fitnesses):
        raise IndexError("Population and fitnesses lists must have the same length.")

    # Check if the population and fitnesses lists are not empty
    if not population or not fitnesses:
        raise ValueError("Population and fitnesses lists cannot be empty.")

    try:
        # Rank the individuals based on their fitness values
        ranks = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

        # Calculate the rank probabilities
        rank_probabilities = [1.0 / (rank + 1) for rank in ranks]
        total_rank_probabilities = sum(rank_probabilities)
        normalized_rank_probabilities = [prob / total_rank_probabilities for prob in rank_probabilities]

        # Select an individual based on the rank probabilities
        selected_index = np.random.choice(len(population), p=normalized_rank_probabilities)
        return population[selected_index]

    except Exception as e:
        print(f"An error occurred: {e}")


def mutate(individual, mutation_rate):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = chr(random.randint(32, 126))
    return ''.join(individual)


def run_first_fit(filename):
    start_time = time.time()
    try:
        problems = read_problems_from_file(filename)
        bin_capacity = 150  # Assuming a fixed bin capacity as it is not provided in the file

        results = []
        for problem_id, items in list(problems.items())[:5]:  # Convert dictionary items to a list before slicing
            num_bins = first_fit(bin_capacity, items)
            results.append((problem_id, num_bins))

            print(f"Problem ID: {problem_id}, Number of bins used: {num_bins}")

        return results
    except Exception as e:
        print(f"An error occurred: {e}")

def first_fit(bin_capacity, items):
    try:
        bins = []

        for item in items:
            placed = False

            for b in bins:
                if b + item <= bin_capacity:
                    bins[bins.index(b)] += item
                    placed = True
                    break

            if not placed:
                bins.append(item)
        end_time = time.time()
        return len(bins)
    except Exception as e:
        print(f"An error occurred: {e}")


def read_problems_from_file(file_path):
    problems = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_problem = None
        for line in lines:
            line = line.strip()
            if line.startswith('u'):
                current_problem = line
                problems[current_problem] = []
            elif current_problem is not None:
                try:
                    item = int(line)
                    problems[current_problem].append(item)
                except ValueError:
                    pass
    return problems



def run_selected_genetic_algorithm():
    print("What do you wish to run?")
    print("1. GA with option of SUS/RWS/TOURNAMENT/RANK")
    print("2. aging")
    print("3. aging with option of SUS/RWS/TOURNAMENT/RANK")
    print("4. Sudoku solver")
    print("5. Bin packing")
    print("6. Bin packing with First Fit algorithm")
    print("0. Quit")

    choice = input("Enter your choice (0, 1, 2, 3, 4, 5): ")

    pop_size = 100
    num_genes = 13
    fitness_func = fitness_GA
    max_generations = 100
    mutation_rate = 0.01

    if choice == '0':
        return
    elif choice == "1":
        use_aging = False
        print("which selection would you like?")
        print("1. SUS")
        print("2. RWS")
        print("3. TOURNAMENT")
        print("4. RANK")
        print("There is no other way... the other way was presented in 1A")
        selection = input("Enter your choice (1, 2, 3, 4): ")
        best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(pop_size, num_genes,
                                                                                               fitness_func,
                                                                                               max_generations,
                                                                                               mutation_rate, selection,
                                                                                               use_aging)
    elif choice == "2":
        use_aging = True
        selection = None
        best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(pop_size, num_genes,
                                                                                               fitness_func,
                                                                                               max_generations,
                                                                                               mutation_rate, selection,
                                                                                               use_aging)
    elif choice == "3":
        use_aging = True
        print("which selection would you like?")
        print("1. SUS")
        print("2. RWS")
        print("3. TOURNAMENT")
        print("4. RANK")
        print("There is no other way... the other way was presented in 1A")
        selection = input("Enter your choice (1, 2, 3, 4): ")
        best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(pop_size, num_genes,
                                                                                               fitness_func,
                                                                                               max_generations,
                                                                                               mutation_rate, selection,
                                                                                               use_aging)
    elif choice == '4':
        sudoku_solver.solve()

    elif choice == '5':
        print("Do you wish to use adaptive fitness and not the fixed fitness?")
        adaptive = input("'y' for adaptive fitness and 'n' for no adaptive fitness: ")

        if adaptive == "y":
            adaptive = True
        else:
            adaptive = False

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'binpack1.txt')

        problems = read_problems_from_file(file_path)

        for problem_id, items in list(problems.items())[:5]:
            bin_capacity = 150

            population_size = 100
            max_generations = 1000
            mutation_rate = 0.01

            print(f"Running genetic algorithm for problem: {problem_id}")
            best_solution, num_bins_used = genetic_algorithm_bin_packing(items, bin_capacity, population_size,
                                                                         max_generations, mutation_rate, adaptive)
            print(f"Best solution for {problem_id} uses {num_bins_used} bins")

    elif choice == '6':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'binpack1.txt')

        run_first_fit(file_path)

    if choice == "5" or choice == "6":
        return
    else:
        print("Best individual:", ''.join(best_individual))
        print("Best fitness:", best_fitness)

if __name__ == "__main__":
    run_selected_genetic_algorithm()
