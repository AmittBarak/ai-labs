import concurrent.futures
from engine.algorithm import GeneticSettings, SelectionMethod, run_genetic_algorithm
from sudoku import utils
from sudoku.crossover import cycle_crossover_2d, pmx_2d
from sudoku.fitness import calculate_sudoku_fitness
from sudoku.genes import individual_generator
from sudoku.mutations import invert_mutation_sudoku, scramble_mutation_sudoku, swap_mutation_sudoku
from sudoku.dataset import games, GAME_SOLUTIONS
from sudoku.utils import calculate_correctness

# List of different parameters to test
mutation_rates = [0.01, 0.02, 0.03, 0.04, 0.05]
population_sizes = [500, 1000, 1500, 2000, 2500, 3000]
max_generations = [250, 500, 750, 1000, 1250, 1500]
crossover_methods = [cycle_crossover_2d, pmx_2d]
mutation_methods = [invert_mutation_sudoku, scramble_mutation_sudoku, swap_mutation_sudoku]
selection_methods = [SelectionMethod.SUS, SelectionMethod.RWS, SelectionMethod.TOURNAMENT, SelectionMethod.RANK]
elitism_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
aging_options = [False, True]

# Function to run the genetic algorithm with a given game index and various parameters
def run_test(game_index, mutation_rate, population_size, max_gen, crossover_method, mutation_method, selection_method, elitism_size, use_aging) -> float:
    selected_game = games[game_index]
    game_solution = GAME_SOLUTIONS[game_index]
    best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
        GeneticSettings(
            population_size=population_size,
            genes_count=81,
            elite_size=elitism_size,
            max_generations=max_gen,
            mutation_rate=mutation_rate,
            selection=selection_method,
            use_aging=use_aging,
            print_function=utils.print_pretty_grid,
            verbose=False,
            crossover_generator=crossover_method,
            mutation_generator=mutation_method(selected_game),
            fitness_calculator=calculate_sudoku_fitness(selected_game),
            individual_generator=individual_generator(selected_game),
            stop_condition_function=utils.sudoku_stop_condition(game_solution=game_solution),
        ),
    )
    correctness = calculate_correctness(best_individual, game_solution)
    utils.print_pretty_grid_diff(best_individual, game_solution)
    print(f"\nGame index: {game_index}, Mutation Rate: {mutation_rate}, Population Size: {population_size}, Max Generations: {max_gen}, Crossover Method: {crossover_method.__name__}, Mutation Method: {mutation_method.__name__}, Selection Method: {selection_method}, Elitism Size: {elitism_size}, Aging: {use_aging}")
    print(f"\nGame index: {game_index}, Correctness: {correctness}%")
    return correctness

# Run tests for each game and parameter
def run_tests_for_game(game_index):
    results = []
    for mutation_rate in mutation_rates:
        for population_size in population_sizes:
            for max_gen in max_generations:
                for crossover_method in crossover_methods:
                    for mutation_method in mutation_methods:
                        for selection_method in selection_methods:
                            for elitism_size in elitism_sizes:
                                for use_aging in aging_options:
                                    correctness = run_test(game_index, mutation_rate, population_size, max_gen, crossover_method, mutation_method, selection_method, elitism_size, use_aging)
                                    results.append((mutation_rate, population_size, max_gen, crossover_method.__name__, mutation_method.__name__, selection_method, elitism_size, use_aging, correctness))
                                    if correctness == 100.0:
                                        print(f"\nGame {game_index} has been solved with 100% correctness!")
                                        return game_index, results
    return game_index, results

if __name__ == "__main__":
    all_results = {game_index: [] for game_index in range(2, 6)}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_tests_for_game, game_index) for game_index in range(2, 6)]
        for future in concurrent.futures.as_completed(futures):
            game_index, results = future.result()
            all_results[game_index] = results

    # Print the results
    for game_index, results in all_results.items():
        for result in results:
            mutation_rate, population_size, max_gen, crossover_method, mutation_method, selection_method, elitism_size, use_aging, correctness = result
            print(f"\nGame {game_index}: Mutation Rate: {mutation_rate}, Population Size: {population_size}, Max Generations: {max_gen}, Crossover Method: {crossover_method}, Mutation Method: {mutation_method}, Selection Method: {selection_method}, Elitism Size: {elitism_size}, Aging: {use_aging}, Correctness: {correctness}%")
