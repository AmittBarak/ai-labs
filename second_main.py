import os
import random
from typing import List, Dict
from collections import defaultdict


def create_initial_population(population_size, num_objects):
    population = [[random.randint(0, num_objects - 1) for _ in range(num_objects)] for _ in range(population_size) ]
    return population

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


def tournament(population, fitnesses, k=3):
    selected = []
    for _ in range(len(population) // 2):
        tournament = random.sample(list(zip(population, fitnesses)), k)
        winner = min(tournament, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

# this crossover method is just like single point of crossover
def crossover(parent1, parent2, crossover_rate=0.5):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2


def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutation_index = random.randrange(len(individual))
        individual[mutation_index] = random.randrange(len(individual))

    return individual


def genetic_algorithm(objects, bin_volume, population_size, max_generations, crossover_rate=0.8, mutation_rate=0.1, elites=0.1):
    population = create_initial_population(population_size, len(objects))
    num_elites = int(population_size * elites)

    for generation in range(max_generations):
        fitnesses = [fitness(individual, objects, bin_volume) for individual in population]
        avg_population_fitness = sum(fitnesses) / len(fitnesses)

        # Preserve elites
        sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])
        elites = [individual for individual, _ in sorted_population[:num_elites]]

        new_population = elites[:]
        selected = tournament(population, fitnesses)

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = new_population
        best_solution = min(population, key=lambda x: fitness(x, objects, bin_volume))
        print(f'Generation {generation}, Best solution fitness: {fitness(best_solution, objects, bin_volume)}')

    best_solution = min(population, key=lambda x: fitness(x, objects, bin_volume))
    print('Best solution:', best_solution)
    print('Number of bins used:', fitness(best_solution, objects, bin_volume))

    return best_solution, fitness(best_solution, objects, bin_volume)


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


def main_run():
    # File MOST be in the same folder as the main code is!
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'binpack1.txt')

    # Parse the input file
    problems = parse_input_file(file_path)

    # Run the genetic algorithm for each test problem with dynamic parameters
    for problem in problems[:5]:
        objects = problem['items']
        bins = problem['bin_capacity']

        # Set dynamic parameters
        population_size = len(objects)
        max_generations = bins
        elites = 0.1

        print(f"Running genetic algorithm for problem: {problem['problem_id']}")
        best_solution, num_bins_used = genetic_algorithm(objects, bins, population_size, max_generations, elites=elites)
        print(f"Best solution for {problem['problem_id']} uses {num_bins_used} bins")


if __name__ == "__main__":
    main_run()
