import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import random
import time
from engine.algorithm import GeneticSettings, SelectionMethod, run_genetic_algorithm
from sudoku import utils
from sudoku.crossover import cycle_crossover_2d, pmx_2d
from sudoku.fitness import calculate_sudoku_fitness
from sudoku.genes import individual_generator
from sudoku.mutations import invert_mutation_sudoku, scramble_mutation_sudoku, swap_mutation_sudoku
from sudoku.dataset import games, GAME_SOLUTIONS
from sudoku.utils import calculate_correctness

# List of different mutation rates to test
mutation_rates = [0.01, 0.02, 0.03, 0.04, 0.05]

results = []

# Parameters for the test
SELECTION_METHOD = SelectionMethod.SUS
PARAM_TITLE = "Mutation Rate"

# Function to run the genetic algorithm with a given game index and mutation rate
def run_test(game_index, mutation_rate) -> float:
    selected_game = games[game_index]
    game_solution = GAME_SOLUTIONS[game_index]
    best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
        GeneticSettings(
            population_size=400,
            genes_count=81,
            elite_size=0.5,
            max_generations=200,
            mutation_rate=mutation_rate,
            selection=SELECTION_METHOD,
            use_aging=False,
            print_function=utils.print_pretty_grid,
            verbose=False,
            crossover_generator=cycle_crossover_2d,
            mutation_generator=invert_mutation_sudoku(selected_game),
            fitness_calculator=calculate_sudoku_fitness(selected_game),
            individual_generator=individual_generator(selected_game),
            stop_condition_function=utils.sudoku_stop_condition(game_solution=game_solution),
        ),
    )
    print(f"Game index: {game_index}, Mutation Rate: {mutation_rate}")
    utils.print_pretty_grid_diff(best_individual, game_solution)
    return calculate_correctness(best_individual, game_solution)

# Run tests for each game and parameter
def run_tests_for_game(game_index):
    results = []
    for mutation_rate in mutation_rates:
        percentage_correct = run_test(game_index, mutation_rate)
        results.append((mutation_rate, percentage_correct))
    return game_index, results

if __name__ == "__main__":
    all_results = {game_index: [] for game_index in range(len(games))}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_tests_for_game, game_index) for game_index in range(len(games))]
        for future in concurrent.futures.as_completed(futures):
            game_index, results = future.result()
            all_results[game_index] = results

    plt.figure(figsize=(12, 8))
    for game_index, results in all_results.items():
        mutation_rates_result, percentages_correct = zip(*results)
        plt.plot(mutation_rates_result, percentages_correct, marker='o', label=f'Game {game_index}')

    plt.title(f'Percentage Correct vs. {PARAM_TITLE}')
    plt.xlabel(f'{PARAM_TITLE}')
    plt.ylabel('Percentage Correct')
    plt.legend()
    plt.grid(True)
    plt.show()
