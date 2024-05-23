import random
import time
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
from typing import List, Dict, Tuple
from collections import defaultdict
from second_main import main_run


def create_initial_population(population_size, num_objects):
    population = [[random.randint(0, num_objects - 1) for _ in range(num_objects)] for _ in range(population_size) ]
    return population

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

def mutate_for_bin_packing(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutation = random.randrange(len(individual))
        individual[mutation] = random.randrange(len(individual))
    return individual


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


# this crossover method is just like single point of crossover
def crossover(parent1, parent2, crossover_rate=0.5):
    if len(parent1) != len(parent2):
        raise ValueError("Both parents must have the same length")
    try:
        if random.random() < crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2
    except Exception as e:
        print(f"An error occurred: {e}")


# Fixed fitness function
# Emphasizes the importance of fitness variance and adaptive fitness functions in evolutionary algorithms
# Adaptive fitness function directly aligns with these principles by adapting to the average population fitness,
# thereby managing variance and promoting better solutions over generations.
def fitness_bin_packing(individual, objects, bin_volume):
    bins: Dict[int, int] = defaultdict(int)
    for bin_number, object_volume in zip(individual, objects):
        bins[bin_number] += object_volume

    total_bins = len(bins)
    overflow_penalty = sum(max(0, volume - bin_volume) for volume in bins.values()) * 100

    return total_bins + overflow_penalty

# Self-adaptive fitness functions
# Adjust based on individual and population performance comparing against the average population fitness is a concrete
# implementation of this concept there for suitable for dynamic and evolving problems.
def adaptive_fitness(individual, objects, bin_volume, avg_population_fitness):
    fitness_score = fitness(individual, objects, bin_volume)
    adaptive_score = fitness_score - avg_population_fitness
    return adaptive_score


def genetic_algorithm_bin_packing(objects, bin_volume, population_size, max_generations, crossover_rate=0.3, mutation_rate=0.1, elites=0.1, use_aging=False):
    try:
        start_CPU_time = time.process_time()
        population = create_initial_population(population_size, len(objects))
        num_elites = int(population_size * elites)
        ages = [random.random() for _ in range(population_size)]

        for generation in range(max_generations):
            try:
                fitnesses = [fitness_bin_packing(individual, objects, bin_volume) for individual in population]
            except Exception as e:
                print(f"Error in fitness function: {e}")
                return None, None

            if use_aging:
                fitnesses = aging(population, fitnesses, ages)

            avg_population_fitness = sum(fitnesses) / len(fitnesses)

            sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])
            elites = [individual for individual, _ in sorted_population[:num_elites]]

            new_population = elites[:]

            # Using tournament and rank for better accurate result
            try:
                tournament = tournament_for_bin_packing(population, fitnesses)
                rank = [rank_bin_packing(population, fitnesses) for _ in range(len(population) // 2)]
            except Exception as e:
                print(f"Error in selection functions: {e}")
                return None, None

            selected = tournament + rank

            while len(new_population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                try:
                    child1, child2 = crossover(parent1, parent2, crossover_rate)
                    new_population.append(mutate_for_bin_packing(child1, mutation_rate))
                    new_population.append(mutate_for_bin_packing(child2, mutation_rate))
                except Exception as e:
                    print(f"Error in crossover or mutation functions: {e}")
                    return None, None

            population = new_population[:population_size]
            best_solution = min(population, key=lambda x: fitness_bin_packing(x, objects, bin_volume))
            print(f'Generation {generation}, Best solution fitness: {fitness_bin_packing(best_solution, objects, bin_volume)}')

        best_solution = min(population, key=lambda x: fitness_bin_packing(x, objects, bin_volume))
        print('Best solution:', best_solution)
        print('Number of bins used:', fitness_bin_packing(best_solution, objects, bin_volume))

        CPU_convergence = time.process_time() - start_CPU_time
        print(f"CPU Time to convergence: {CPU_convergence:.2f} seconds")

        return best_solution, fitness_bin_packing(best_solution, objects, bin_volume)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def mutate(individual, mutation_rate):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = chr(random.randint(32, 126))
    return ''.join(individual)


def parse_input_file(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()]

    problems = []
    num_problems = int(lines[0])
    idx = 1

    for _ in range(num_problems):
        problem_id = lines[idx]
        idx += 1
        bin_capacity, num_items, _ = map(int, lines[idx].split())
        idx += 1
        items = [int(line) for line in lines[idx:idx + num_items]]
        idx += num_items
        problems.append({'problem_id': problem_id, 'bin_capacity': bin_capacity, 'items': items})

    return problems


def run_selected_genetic_algorithm():
    print("What do you wish to run?")
    print("1. GA with option of SUS/RWS/TOURNAMENT/RANK")
    print("2. aging")
    print("3. aging with option of SUS/RWS/TOURNAMENT/RANK")
    print("4. Sudoku solver")
    print("5. Bin packing")
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
        return
    elif choice == '5':
        print("Do you with to use aging?")
        set_aging = input("'y' for aging and 'n' for no aging: ")
        if set_aging == 'y':
            set_aging = True
        else:
            set_aging = False

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'binpack1.txt')

        problems = parse_input_file(file_path)

        for problem in problems[:5]:
            objects = problem['items']
            bins = problem['bin_capacity']

            population_size = len(objects)
            max_generations = bins
            elites = 0.1

            print(f"Running genetic algorithm for problem: {problem['problem_id']}")
            best_solution, num_bins_used = genetic_algorithm_bin_packing(objects, bins, population_size,
                                                                         max_generations, elites=elites, use_aging=set_aging)
            print(f"Best solution for {problem['problem_id']} uses {num_bins_used} bins")

    if choice == "5":
        return
    else:
        print("Best individual:", ''.join(best_individual))
        print("Best fitness:", best_fitness)

if __name__ == "__main__":
    run_selected_genetic_algorithm()
