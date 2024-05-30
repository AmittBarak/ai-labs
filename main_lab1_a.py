import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import collections

def single_point_crossover(parent1, parent2):
    """
    Performs single-point crossover on two parent lists.

    Args:
        parent1 (list): The first parent list.
        parent2 (list): The second parent list.

    Returns:
        list: The child list created by performing single-point crossover.

    Raises:
        ValueError: If the parent lists have different lengths.

    This function performs single-point crossover on two parent lists. A random crossover point is chosen, and the child
    is created by taking the first part of `parent1` up to the crossover point and the remaining part from `parent2`.
    If the parent lists have different lengths, a `ValueError` is raised.
    """
    try:
        if len(parent1) != len(parent2):
            raise ValueError("Parent lists must have the same length.")

        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    except Exception as e:
        print(f"Error in single_point_crossover: {e}")
        raise

def two_point_crossover(parent1, parent2):
    """
    Performs two-point crossover on two parent strings.

    Args:
        parent1 (str): The first parent string.
        parent2 (str): The second parent string.

    Returns:
        str: The child string created by performing two-point crossover.

    Raises:
        ValueError: If the parent strings have different lengths.

    This function performs two-point crossover on two parent strings. Two random crossover points are chosen, and the
    child is created by taking the first part of `parent1` up to the first crossover point, the middle part from `parent2`
    between the two crossover points, and the remaining part from `parent1` after the second crossover point. If the
    parent strings have different lengths, a `ValueError` is raised.
    """
    try:
        if len(parent1) != len(parent2):
            raise ValueError("Parent strings must have the same length.")

        ind1, ind2 = sorted(random.sample(range(1, len(parent1)), 2))

        child = parent1[:ind1] + parent2[ind1:ind2] + parent1[ind2:]
        return child
    except Exception as e:
        print(f"Error in two_point_crossover: {e}")
        raise


def chartMaker(all_fitness_scores, all_generations):
    """
    Creates a scatter plot with vertical lines to visualize the normalized fitness scores across generations.

    Args:
        all_fitness_scores (list): A list of lists, where each sublist contains the fitness scores for a generation.
        all_generations (list): A list of generation numbers corresponding to the fitness scores.

    Returns:
        Chart

    This function creates a scatter plot with vertical lines to visualize the normalized fitness scores across generations.
    The scores are normalized to a range of 0 to 100 for each generation. Vertical lines are drawn in red to indicate the
    range of scores for each generation, and the normalized scores are plotted as blue dots. The x-axis represents the
    generation number, and the y-axis represents the normalized fitness score. The plot is displayed using `plt.show()`.
    """
    try:
        plt.figure(figsize=(15, 5))

        # Hold the x and y values for the scatter.
        all_x = []
        all_y = []
        # Hold the x and y values for the line segments.
        all_lines_x = []
        all_lines_y = []

        # Loop each generation and its corresponding fitness scores.
        for generation, fitness_scores in zip(all_generations, all_fitness_scores):
            # Determine the maximum and minimum scores, use 0 if there are no scores at all.
            max_score = max(fitness_scores) if fitness_scores else 0
            min_score = min(fitness_scores) if fitness_scores else 0
            # Normalize scores, avoiding division by zero otherwise the mathematical world will collapse.
            normalized_scores = [0 if max_score == min_score else ((score - min_score) / (max_score - min_score) * 100)
                             for score in fitness_scores]

            # Append data points for each normalized score to the lists.
            for score in normalized_scores:
                all_x.append(generation)
                all_y.append(score)
                all_lines_x.extend([generation, generation, None])
                all_lines_y.extend([0, score, None])

        # Draw vertical lines in red to indicate the scores for each generation.
        plt.plot(all_lines_x, all_lines_y, color='red')
        # Draw normalized scores as blue dots.
        plt.scatter(all_x, all_y, color='blue')

        # Setting up axes, titles, and labels.
        max_generation = max(all_generations)
        generation_bins = np.arange(0, max_generation + 11, 10)  # Create bins for x-axis in jumps of 10 to create a separation.
        plt.title('Normalized Fitness Scores By Generation')
        plt.xlabel('Generation')
        plt.ylabel('Normalized Fitness')
        plt.grid(True)
        plt.xticks(generation_bins)

        # Display the plot.
        plt.show()
    except Exception as e:
        print(f"Error in chart maker: {e}")
        raise


def fitness_bullseye(individual):
    """
    Evaluates the fitness of an individual against a target string.

    This function calculates the fitness score of an individual by comparing it to a target string.
    The fitness score is calculated as follows:

    1. The number of exact character matches in the same position is counted.
    2. For each character in the individual, a bonus of 0.5 is added for each occurrence of that character in the target string,
       up to the maximum number of occurrences in the target string.

    The final fitness score is the sum of the exact matches and the bonus points.

    Args:
        individual (str): The individual string to be evaluated.

    Returns:
        float: The fitness score of the individual.

    Raises:
        Exception: If an error occurs during the fitness calculation.
    """
    target = "Hello, world!"
    try:
        # Calculate points for matches
        score = sum(a == b for a, b in zip(individual, target))

        # Calculate bonus points for matching characters, regardless of position
        target_counter = collections.Counter(target)
        bonus = sum(min(count, target_counter[char]) * 0.5 for char, count in collections.Counter(individual).items())

        return score + bonus
    except Exception as e:
        print(f"Error in fitness_bullseye: {e}")
        raise

# Added to this code after it was missing from the original given code
def mutate(individual, mutation_rate):
    try:
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = chr(random.randint(32, 126))
        return individual
    except Exception as e:
        print(f"Error in mutate: {e}")
        raise

# Define the fitness function
def fitness(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score


# Define the genetic algorithm
def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, mutation_rate, selection):
    start_CPU_time = time.process_time()
    start_time = time.time()

    all_fitness_scores = []
    all_generations = []
    population = [[''.join(chr(random.randint(32, 126))) for _ in range(num_genes)] for _ in range(pop_size)]

    ages = [random.random() for _ in range(pop_size)]

    for generation in range(max_generations):
        generation_CPU_start = time.process_time()
        generation_start = time.time()

        fitnesses = [fitness_func(individual) for individual in population]
        all_fitness_scores.append(fitnesses)
        all_generations.append(generation)

        avg = np.mean(fitnesses)
        dev = np.std(fitnesses)
        print(f"Generation {generation}: Average Fitness = {int(round(avg))}, Standard Deviation = {int(round(dev))}")

        elite_size = int(pop_size * 0.1)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        offspring = []
        while len(offspring) < pop_size - elite_size:
            parent1, parent2 = random.sample(elites, 2)
            if selection == "single":
                child = single_point_crossover(parent1, parent2)
            elif selection == "two":
                child = two_point_crossover(parent1, parent2)
            else:
                child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(num_genes)]
            child = mutate(child, mutation_rate)
            offspring.append(child)
        population = elites + offspring

        generation_CPU_end = time.process_time()
        generation_CPU_ticks = generation_CPU_end - generation_CPU_start
        generation_CPU_elapsed = generation_CPU_end - start_CPU_time
        generation_end = time.time()
        generation_ticks = generation_end - generation_start
        generation_elapsed = generation_end - start_time

        print(f"CPU Generation {generation}: Ticks Clock CPU = {generation_CPU_ticks} seconds, "
              f"Total Elapsed = {generation_CPU_elapsed:.2f} seconds")
        print(f"Time Generation {generation}: Ticks Clock = {generation_ticks} seconds, "
              f"Total Elapsed = {generation_elapsed:.2f} seconds")

    convergence = time.time() - start_time
    CPU_convergence = time.process_time() - start_CPU_time
    print(f"CPU Time to convergence: {CPU_convergence:.2f} seconds")
    print(f"Time to convergence: {convergence:.2f} seconds")

    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    return best_individual, best_fitness, all_fitness_scores, all_generations


""" Old running "main" 
# Run the genetic algorithm and print the result
best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(pop_size=100, num_genes=13, fitness_func=fitness, max_generations=100)
print("Best individual:", ''.join(best_individual))
print("Best fitness:", best_fitness)
# Create's a chart
chartMaker(all_fitness_scores, all_generations)
"""


def run_selected_genetic_algorithm():
    """
    This function allows the user to select a genetic algorithm to run and optionally display a chart.
    It asks the user for their choice, and then calls the genetic_algorithm function with the appropriate
    parameters based on the user's selection. Finally, it prints the best individual and fitness score,
    and optionally displays a chart of the fitness scores across generations.
    """
    # Displaying the choices
    print("Welcome to the genetics olympics")
    print("Select the genetic algorithm to run:")
    print("1. Single-point crossover")
    print("2. Two-point crossover")
    print("3. Uniform genetic algorithm")
    print("4. Single-point crossover with bullseye heuristic")
    print("5. Two-point crossover with bullseye heuristic")
    print("6. Uniform genetic algorithm with bullseye heuristic")
    print("0. To quit")

    choice = input("Enter your choice (0, 1, 2, 3, 4, 5, 6): ")
    if choice == '0':
        return

    print("Do you wish to see a chart?")
    chart = input("'y' to see a chart or 'n' not to see a chart ")

    # Parameters for genetic algorithms same has Shay defined
    pop_size = 100
    num_genes = 13
    fitness_func = fitness
    max_generations = 100
    better_fitness = fitness_bullseye
    mutation_rate = 0.01
    selection = None

    try:
        if choice == '1':
            selection = "single"
            best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(
                pop_size, num_genes, fitness_func, max_generations, mutation_rate, selection)
        elif choice == '2':
            selection = "two"
            best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(
                pop_size, num_genes, fitness_func, max_generations, mutation_rate, selection)
        elif choice == '3':
            best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(
                pop_size, num_genes, fitness_func, max_generations, mutation_rate, selection)
        elif choice == '4':
            selection = "single"
            best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(
                pop_size, num_genes, better_fitness, max_generations, mutation_rate, selection)
        elif choice == '5':
            selection = "two"
            best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(
                pop_size, num_genes, better_fitness, max_generations, mutation_rate, selection)
        elif choice == '6':
            best_individual, best_fitness, all_fitness_scores, all_generations = genetic_algorithm(
                pop_size, num_genes, better_fitness, max_generations, mutation_rate, selection)
        else:
            print("Invalid choice. Try again.")
            return run_selected_genetic_algorithm()
    except Exception as e:
        print(f"An error occurred: {e}")
        return run_selected_genetic_algorithm()

    # Results
    print("Best individual:", ''.join(best_individual))
    print("Best fitness:", best_fitness)
    if chart == 'y':
        chartMaker(all_fitness_scores, all_generations)


if __name__ == "__main__":
    run_selected_genetic_algorithm()