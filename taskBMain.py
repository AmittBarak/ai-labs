import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import collections
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
    young_age_threshold = 0.25
    adult_age_threshold = 0.75
    young_scaling_factor = 0.5
    adult_scaling_factor = 1.5
    old_scaling_factor = 0.5

    aged_fitnesses = []
    for age, fitness in zip(age_distribution, fitnesses):
        if age < young_age_threshold:
            aged_fitness = fitness * young_scaling_factor
        elif age < adult_age_threshold:
            aged_fitness = fitness * adult_scaling_factor
        else:
            aged_fitness = fitness * old_scaling_factor
        aged_fitnesses.append(aged_fitness)

    return aged_fitnesses


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

# Fixed fitness function
# Emphasizes the importance of fitness variance and adaptive fitness functions in evolutionary algorithms
# Adaptive fitness function directly aligns with these principles by adapting to the average population fitness,
# thereby managing variance and promoting better solutions over generations.
def fitness(individual, objects, bin_volume):
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

def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, mutation_rate, selection, use_aging):
    start_CPU_time = time.process_time()
    all_fitness_scores = []
    all_generations = []
    population = [[''.join(chr(random.randint(32, 126)) for _ in range(num_genes))] for _ in range(pop_size)]
    ages = [random.random() for _ in range(pop_size)]

    for generation in range(max_generations):
        generation_CPU_start = time.process_time()
        fitnesses = [fitness_func(individual[0]) for individual in population]
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

def mutate(individual, mutation_rate):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = chr(random.randint(32, 126))
    return ''.join(individual)


def parse_input_file(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    num_problems = int(lines[0])
    problems = []
    line_idx = 1

    for _ in range(num_problems):
        problem_id = lines[line_idx]
        line_idx += 1
        bin_capacity, num_items, _ = map(int, lines[line_idx].split())
        line_idx += 1
        items = [int(line) for line in lines[line_idx:line_idx + num_items]]
        line_idx += num_items

        problems.append({
            'problem_id': problem_id,
            'bin_capacity': bin_capacity,
            'items': items
        })

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
        return main_run()

    print("Best individual:", ''.join(best_individual))
    print("Best fitness:", best_fitness)

if __name__ == "__main__":
    run_selected_genetic_algorithm()