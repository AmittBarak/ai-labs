import os
import random
import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np

import sudoku.utils
from engine.selection import nieching_partition, crowding_density, species_speciation
from ga_gep.ga import genetic_algorithm, print_tree, tree_to_expression, fitness_ga
from ga_gep.gep import gep_algorithm, gep_to_expression, fitness_gep_with_bloat_control, GEPChromosome
from hello_world.hello_world import crossover, fitness_GA, mutate
from bin_packing import utils as bin_packing_utils
from bin_packing import bin_packing as bin_packing
from engine.algorithm import GeneticSettings, SelectionMethod, run_genetic_algorithm
from hillis_exp.Hillis import CoevolutionaryAlgorithm
from lab3.CVRP import ackley_function, flatten_routes, aco_solution, sa_solution, ils_solution, ts_solution, \
    ga_solution, alns_solution, aco_length, coords, print_routes, plot_routes, sa_length, ils_length, ts_length, \
    ga_length, aco
from sudoku import utils
from sudoku.crossover import cycle_crossover_2d
from sudoku.fitness import calculate_sudoku_fitness
from sudoku.genes import individual_generator
from sudoku.mutations import invert_mutation_sudoku
from sudoku.dataset import games, GAME_SOLUTIONS
from bin_packing.bin_packing import MutationOperators
from baldwins_exp.baldwin import EvolutionaryAlgorithm
import matplotlib.pyplot as plt
import concurrent.futures


def main():
    """Main function to run the selected genetic algorithm."""
    display_menu()
    # choice = input("Enter your choice (0, 1, 2, 3, 4, 5, 6, 7, 8): ")
    choice = input("Enter your choice (0, 1, 2, 3, 4, 5, 6, 7, 8): ")

    if choice == '0':
        quit()

    options = {
        # '1': run_ga_with_selection_options,
        # '2': run_aging,
        # '3': run_aging_with_selection_options,
        # '4': run_sudoku_solver,
        '1': run_bin_packing,
        '2': run_bin_packing_first_fit,
        '3': run_bin_packing_with_crowding_density_nieching_partition_species_speciation,
        '4': run_bin_packing_with_mutation_function,
        '5': run_bladwins_exp,
        '6': run_cvrp,
        '7': run_hillis,
        '8': run_ga_gep
    }

    if choice in options:
        options[choice]()
    else:
        print("Invalid choice, please try again.")
        main()


def display_menu():
    """Display the main menu options."""
    print("What do you wish to run?")
    # print("1. GA with option of SUS + Linear scaling/RWS + Linear scaling /TOURNAMENT/RWS + RANK")
    # print("2. Aging")
    # print("3. Aging with option of SUS + Linear scaling/RWS + Linear scaling/TOURNAMENT/RWS + RANK")
    # print("4. Sudoku solver")
    print("1. Bin packing")
    print("2. Bin packing with First Fit algorithm")
    print("3. Bin packing with crowding density/nieching_partition/species_speciation")
    print("4. Bin packing with mutations")
    print("5. Baldwin's experiment")
    print("6. CVRP")
    print("7. Hillis")
    print("8. GA and GEP")
    print("0. Quit")


def run_ga_with_selection_options():
    """Run GA with different selection methods."""
    best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
        GeneticSettings(
            use_aging=False,
            genes_count=13,
            population_size=400,
            max_generations=100,
            mutation_rate=0.04,
            selection=get_selection_method(),
            fitness_calculator=fitness_GA,
            individual_generator=lambda genes: ''.join(chr(random.randint(32, 126)) for _ in range(genes)),
            mutation_generator=mutate,
            crossover_generator=crossover
        )
    )

    print("Best individual:", best_individual)
    print("Best fitness:", best_fitness)


def run_aging():
    """Run GA with aging."""
    best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
        GeneticSettings(
            use_aging=True,
            genes_count=13,
            population_size=100,
            max_generations=100,
            mutation_rate=0.01,
            selection=SelectionMethod.NO_SELECTION,
            fitness_calculator=fitness_GA,
            individual_generator=lambda genes: ''.join(chr(random.randint(32, 126)) for _ in range(genes)),
            mutation_generator=mutate,
            crossover_generator=crossover
        )
    )
    print("Best individual:", best_individual)
    print("Best fitness:", best_fitness)


def run_ga_gep():
    print("Part 1: Genetic Algorithm for Boolean Function Optimization")

    boolean_functions = {
        "XOR": lambda a, b: a ^ b,
        "AND": lambda a, b: a and b,
        "OR": lambda a, b: a or b
    }

    for func_name, func in boolean_functions.items():
        best_solution = genetic_algorithm(func)
        print(f"\nBest solution for {func_name} found:")
        print("Tree representation:")
        print_tree(best_solution)
        print("\nExpression:")
        print(tree_to_expression(best_solution))
        print(f"Fitness: {fitness_ga(best_solution, func)}")

    print("\nPart 2: Gene Expression Programming for Polynomial Fitting")
    target_data = [(x, x ** 3 + 2 * x ** 2 + x + 1) for x in [-1, -0.5, 0, 0.5, 1]]

    best_solution_gep = gep_algorithm(target_data)
    print_gep_solution(best_solution_gep, target_data)

    print("\nPart 3: Univariate Polynomial Fitting")
    univariate_data = [(-1, 1), (-0.5, 0.75), (0, 1), (0.5, 1.75), (1, 3)]
    best_univariate_gep = gep_algorithm(univariate_data)
    print_gep_solution(best_univariate_gep, univariate_data, "univariate polynomial")



def print_gep_solution(solution: GEPChromosome, data: List[Tuple[float, float]], description: str = ""):
    print(f"Best {description} solution found:")
    print(f"Genes: {solution.genes}")
    print(f"Expression: {gep_to_expression(solution)}")
    print(f"Fitness: {fitness_gep_with_bloat_control(solution, data)}")


def run_aging_with_selection_options():
    """Run GA with aging and different selection methods."""
    best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
        GeneticSettings(
            use_aging=True,
            genes_count=13,
            population_size=100,
            max_generations=100,
            mutation_rate=0.01,
            selection=get_selection_method(),
            fitness_calculator=fitness_GA,
            individual_generator=lambda genes: ''.join(chr(random.randint(32, 126)) for _ in range(genes)),
            mutation_generator=mutate,
            crossover_generator=crossover
        )
    )
    print("Best individual:", best_individual)
    print("Best fitness:", best_fitness)


def run_sudoku_solver():
    """Run GA for Sudoku solving."""
    game_settings = {
        0: {'population_size': 1000, 'mutation_rate': 0.04, 'selection': SelectionMethod.RWS, 'use_aging': True,
            "elite_size": 0.4, "max_generations": 500, "mutation_generator": invert_mutation_sudoku(games[0])},

        1: {'population_size': 1000, 'mutation_rate': 0.04, 'selection': SelectionMethod.RWS, 'use_aging': True,
            "elite_size": 0.4, "max_generations": 500, "mutation_generator": invert_mutation_sudoku(games[1])},

        2: {'population_size': 10000, 'mutation_rate': 0.03, 'selection': SelectionMethod.RWS, 'use_aging': True,
            "elite_size": 0.4, "max_generations": 5000, "mutation_generator": invert_mutation_sudoku(games[2])},

        3: {'population_size': 10000, 'mutation_rate': 0.04, 'selection': SelectionMethod.RWS, 'use_aging': True,
            "elite_size": 0.4, "max_generations": 5000, "mutation_generator": invert_mutation_sudoku(games[3])},

        4: {'population_size': 10000, 'mutation_rate': 0.04, 'selection': SelectionMethod.RWS, 'use_aging': True,
            "elite_size": 0.4, "max_generations": 5000, "mutation_generator": invert_mutation_sudoku(games[4])},

        5: {'population_size': 10000, 'mutation_rate': 0.04, 'selection': SelectionMethod.RWS, 'use_aging': True,
            "elite_size": 0.4, "max_generations": 5000, "mutation_generator": invert_mutation_sudoku(games[5])}
    }

    print("Available Sudoku games: 0, 1, 2, 3, 4, 5")
    selected_games = input("Enter the numbers of the games you want to solve, separated by commas: ")
    selected_games = [int(x) for x in selected_games.split(",") if x.isdigit() and 0 <= int(x) <= 5]

    def solve_game(game_index):
        chosen_game = games[game_index]
        settings = game_settings.get(game_index, game_settings[0])  # Default to game 0 settings if not found
        best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
            GeneticSettings(
                population_size=settings['population_size'],
                genes_count=81,
                elite_size=settings['elite_size'],
                max_generations=settings['max_generations'],
                mutation_rate=settings['mutation_rate'],
                selection=settings['selection'],
                use_aging=settings['use_aging'],
                print_function=utils.print_pretty_grid,
                verbose=True,
                crossover_generator=cycle_crossover_2d,
                mutation_generator=settings['mutation_generator'],
                fitness_calculator=calculate_sudoku_fitness(chosen_game),
                individual_generator=individual_generator(chosen_game),
                adjust_parameters=sudoku.utils.adjust_parameters,  # Enable dynamic parameter adjustment
                stop_condition_function=utils.sudoku_stop_condition(game_solution=GAME_SOLUTIONS[game_index])
            ),
        )
        return best_individual

    solutions = []
    for game_index in selected_games:
        try:
            solution = solve_game(game_index)
            solutions.append((game_index, solution))
        except Exception as e:
            print(f"Game {game_index} generated an exception: {e}")

    for game_index, solution in solutions:
        print(f"Solution for Game {game_index + 1}:")
        utils.print_pretty_grid_diff(solution, GAME_SOLUTIONS[game_index])
        print(f"Is valid: {utils.is_valid_sudoku(solution)}")


def run_bin_packing():
    """Run GA for bin packing."""
    adaptive = input("'y' for adaptive fitness and 'n' for no adaptive fitness: ").lower() == 'y'
    use_aging = input("'y' for aging and 'n' for no aging: ").lower() == 'y'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'binpack1.txt')
    problems = bin_packing_utils.read_problems_from_file(file_path)
    for problem_id, items in list(problems.items())[:5]:
        bin_capacity = 150
        population_size = 100
        max_generations = 300
        mutation_rate = 0.01
        print(f"Running genetic algorithm for problem: {problem_id}")
        ga_bin_packing = bin_packing.GeneticAlgorithmBinPacking(
            items, bin_capacity, population_size, max_generations, mutation_rate, adaptive, use_aging)
        best_solution, num_bins_used, best_generation, ga_time = ga_bin_packing.run()
        print(f"Best solution for {problem_id} uses {num_bins_used} bins and the number of generations it took {best_generation}")


def run_bin_packing_first_fit():
    """Run bin packing with First Fit algorithm."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'binpack1.txt')
    bin_packing_utils.run_first_fit(file_path)

def run_bin_packing_with_crowding_density_nieching_partition_species_speciation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'binpack1.txt')

    problems = bin_packing_utils.read_problems_from_file(file_path)
    print("Select the function to use:")
    print("1. Crowding density")
    print("2. Nieching partition")
    print("3. Species speciation")
    function_choice = input("Enter your choice (1, 2, 3): ")

    function_map = {
        "1": crowding_density,
        "2": nieching_partition,
        "3": species_speciation
    }

    selected_function = function_map.get(function_choice)
    if selected_function is None:
        print("Invalid choice. Exiting.")
        return

    adaptive = input("'y' for adaptive fitness and 'n' for no adaptive fitness: ").lower() == 'y'
    use_aging = input("'y' for aging and 'n' for no aging: ").lower() == 'y'

    for problem_id, items in list(problems.items())[:5]:
        bin_capacity = 150
        population_size = 100
        max_generations = 300
        mutation_rate = 0.01
        print(f"Running genetic algorithm for problem: {problem_id}")
        ga_bin_packing = bin_packing.GeneticAlgorithmBinPacking(
            items, bin_capacity, population_size, max_generations, mutation_rate, adaptive, use_aging,
            binning_function=selected_function
        )
        best_solution, num_bins_used, best_generation = ga_bin_packing.run()
        print(f"Best solution for {problem_id} uses {num_bins_used} bins and took {best_generation} generations.")


def run_bin_packing_with_mutation_function():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'binpack1.txt')

    problems = bin_packing_utils.read_problems_from_file(file_path)
    print("Select the function to use:")
    print("1. basic")
    print("2. non_uniform")
    print("3. adaptive")
    print("4. triggered_hyper")
    print("5. self_adaptive")
    function_choice = input("Enter your choice (1, 2, 3, 4, 5): ")

    function_map = {
        "1": MutationOperators.basic_mutation,
        "2": MutationOperators.non_uniform_mutation,
        "3": MutationOperators.adaptive_mutation,
        "4": MutationOperators.triggered_hyper_mutation,
        "5": MutationOperators.self_adaptive_mutation
    }

    selected_function = function_map.get(function_choice)
    if selected_function is None:
        print("Invalid choice. Exiting.")
        return

    adaptive = False
    use_aging = False

    for problem_id, items in list(problems.items())[:5]:
        bin_capacity = 150
        population_size = 100
        max_generations = 300
        mutation_rate = 0.01
        print(f"Running genetic algorithm for problem: {problem_id}")
        ga_bin_packing = bin_packing.GeneticAlgorithmBinPacking(
            items, bin_capacity, population_size, max_generations, mutation_rate, adaptive, use_aging,
            None, mutation_function=selected_function)
        best_solution, num_bins_used, best_generation = ga_bin_packing.run()
        print(f"Best solution for {problem_id} uses {num_bins_used} bins and took {best_generation} generations.")

def run_bladwins_exp():
    target_pattern = ['1', '0', '{', '1', '}']
    generations = 100
    mutation_rate = 0.01

    # Run simulation with learning
    ea_with_learning = EvolutionaryAlgorithm(target_pattern, generations=generations, mutation_rate=mutation_rate)
    correct_matches_with_learning, incorrect_positions_with_learning, learned_bits_with_learning = ea_with_learning.run_simulation(
        with_learning=True)

    # Run simulation without learning
    ea_without_learning = EvolutionaryAlgorithm(target_pattern, generations=generations, mutation_rate=mutation_rate)
    correct_matches_without_learning, incorrect_positions_without_learning, learned_bits_without_learning = ea_without_learning.run_simulation(
        with_learning=False)

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(correct_matches_with_learning, label='Correct Matches with Learning')
    plt.plot(correct_matches_without_learning, label='Correct Matches without Learning')
    plt.xlabel('Generation')
    plt.ylabel('Percentage')
    plt.title('Correct Matches')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(incorrect_positions_with_learning, label='Incorrect Positions with Learning')
    plt.plot(incorrect_positions_without_learning, label='Incorrect Positions without Learning')
    plt.xlabel('Generation')
    plt.ylabel('Percentage')
    plt.title('Incorrect Positions')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.plot(learned_bits_with_learning, label='Bits Learned with Learning')
    plt.plot(learned_bits_without_learning, label='Bits Learned without Learning')
    plt.xlabel('Generation')
    plt.ylabel('Percentage')
    plt.title('Bits Learned')
    plt.legend()

    plt.show()

def run_cvrp():
    print("Testing on Ackley Function:")
    aco_ackley = ackley_function(np.array(flatten_routes(aco_solution)))
    sa_ackley = ackley_function(np.array(flatten_routes(sa_solution)))
    ils_ackley = ackley_function(np.array(flatten_routes(ils_solution)))
    ts_ackley = ackley_function(np.array(flatten_routes(ts_solution)))
    ga_ackley = ackley_function(np.array(flatten_routes([ga_solution])))
    alns_ackley = ackley_function(np.array(flatten_routes(alns_solution)))

    print_routes(aco_solution, aco_length, "ACO")
    plot_routes(aco_solution, coords, "ACO")
    print(f"Ackley Value: {aco_ackley}")

    print_routes(sa_solution, sa_length, "SA")
    plot_routes(sa_solution, coords, "SA")
    print(f"Ackley Value: {sa_ackley}")

    print_routes(ils_solution, ils_length, "ILS")
    plot_routes(ils_solution, coords, "ILS")
    print(f"Ackley Value: {ils_ackley}")

    print_routes(ts_solution, ts_length, "TS")
    plot_routes(ts_solution, coords, "TS")
    print(f"Ackley Value: {ts_ackley}")

    print_routes([ga_solution], ga_length, "GA Island Model")
    plot_routes([ga_solution], coords, "GA Island Model")
    print(f"GA Island Model Ackley Value: {ga_ackley}")

    print_routes(alns_solution, sum(aco.route_length(route) for route in alns_solution), "ALNS")
    plot_routes(alns_solution, coords, "ALNS")
    print(f"Ackley Value: {alns_ackley}")

def run_hillis():
    # Parameters for K=6
    vector_length = 6
    population_size = 100
    num_generations = 100
    mutation_rate = 0.1
    num_offspring = 50

    # Initial populations for K=6
    ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=True, initial_mutation_rate=mutation_rate)
    fitness_history = ca.evolve(num_generations, num_offspring)
    ca.plot_fitness(fitness_history, 6, True)

    # Cross-validation for K=6
    mean_score, std_dev = ca.cross_validate(k=5)
    print(f"Cross-Validation Mean Score for K=6: {mean_score}")
    print(f"Cross-Validation Standard Deviation for K=6: {std_dev}")

    # Best network found for K=6
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=6:")
    print(best_network.network)
    best_network.plot_network(6, True)  # Plot the network visualization

    # Compare with QuickSort for K=6
    correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
    print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
    print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")

    # Parameters for K=10
    vector_length = 10
    population_size = 150
    num_generations = 500
    mutation_rate = 0.1
    num_offspring = 100

    # Initial populations for K=10
    ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=True, initial_mutation_rate=mutation_rate)
    fitness_history = ca.evolve(num_generations, num_offspring)
    ca.plot_fitness(fitness_history, 10, True)

    # Cross-validation for K=10
    mean_score, std_dev = ca.cross_validate(k=5)
    print(f"Cross-Validation Mean Score for K=10: {mean_score}")
    print(f"Cross-Validation Standard Deviation for K=10: {std_dev}")

    # Best network found for K=10
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=10:")
    print(best_network.network)
    best_network.plot_network(10, True)  # Plot the network visualization

    # Compare with QuickSort for K=10
    correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
    print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
    print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")

    # Parameters for K=10
    vector_length = 10
    population_size = 150
    num_generations = 500
    mutation_rate = 0.1
    num_offspring = 100

    # Initial populations for K=10
    print("For K=10 without bitonic")
    ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=False, initial_mutation_rate=mutation_rate)
    fitness_history = ca.evolve(num_generations, num_offspring)
    ca.plot_fitness(fitness_history, 10, False)

    # Cross-validation for K=10
    mean_score, std_dev = ca.cross_validate(k=5)
    print(f"Cross-Validation Mean Score for K=10: {mean_score}")
    print(f"Cross-Validation Standard Deviation for K=10: {std_dev}")

    # Best network found for K=10
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=10:")
    print(best_network.network)
    best_network.plot_network(10, False)  # Plot the network visualization

    # Compare with QuickSort for K=10
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
    ca.plot_fitness(fitness_history, 16, True)

    # Cross-validation for K=16
    mean_score, std_dev = ca.cross_validate(k=5)
    print(f"Cross-Validation Mean Score for K=16: {mean_score}")
    print(f"Cross-Validation Standard Deviation for K=16: {std_dev}")

    # Best network found for K=16
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=16")
    print(best_network.network)
    best_network.plot_network(16, True)  # Plot the network visualization

    # Compare with QuickSort for K=16
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
    print("for K=16 without bitonic")
    ca = CoevolutionaryAlgorithm(population_size, vector_length, use_bitonic=False, initial_mutation_rate=mutation_rate)
    fitness_history = ca.evolve(num_generations, num_offspring)
    ca.plot_fitness(fitness_history, 16, False)

    # Cross-validation for K=16
    mean_score, std_dev = ca.cross_validate(k=5)
    print(f"Cross-Validation Mean Score for K=16: {mean_score}")
    print(f"Cross-Validation Standard Deviation for K=16: {std_dev}")

    # Best network found for K=16
    best_network = max(ca.population, key=lambda network: network.fitness(ca.vector_population))
    print("Best Network Found for K=16")
    print(best_network.network)
    best_network.plot_network(16, False)  # Plot the network visualization

    # Compare with QuickSort for K=16
    correct, quicksort_correct = ca.compare_with_quicksort(best_network, ca.vector_population)
    print(f"Evolved Network Correctness: {correct}/{len(ca.vector_population)}")
    print(f"QuickSort Correctness: {quicksort_correct}/{len(ca.vector_population)}")


def get_selection_method() -> SelectionMethod:
    """Get the selection method from the user."""
    global selection_choice
    valid_selection = False
    while not valid_selection:
        print("Which selection would you like?")
        print("1. SUS + Linear scaling")
        print("2. RWS + Linear scaling")
        print("3. TOURNAMENT")
        print("4. RWS + RANK")
        selection_choice = input("Enter your choice (1, 2, 3, 4): ")
        if selection_choice in ["1", "2", "3", "4"]:
            valid_selection = True
        else:
            print("Invalid choice, please select again.")
    return SelectionMethod(int(selection_choice))


if __name__ == '__main__':
    main()