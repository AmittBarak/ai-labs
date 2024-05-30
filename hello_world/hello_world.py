import collections
import random

import numpy as np
from matplotlib import pyplot as plt


# Define the fitness function for Genetic Algorithm
def fitness_GA(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score


def mutate(individual: any, mutation_rate: float):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = chr(random.randint(32, 126))
    return ''.join(individual)


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
    child is created by taking the first part of `parent1` up to the first crossover point, the middle part from
    `parent2`
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

    This function creates a scatter plot with vertical lines to visualize the normalized fitness scores across
    generations.
    The scores are normalized to a range of 0 to 100 for each generation. Vertical lines are drawn in red to indicate
    the
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
        generation_bins = np.arange(0, max_generation + 11,
                                    10)  # Create bins for x-axis in jumps of 10 to create a separation.
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
    2. For each character in the individual, a bonus of 0.5 is added for each occurrence of that character in the
    target string,
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


def crossover(parent1, parent2):
    return ''.join([parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))])
