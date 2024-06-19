import os
import random
import sudoku.utils
from engine.selection import nieching_partition, crowding_density, species_speciation
from hello_world.hello_world import crossover, fitness_GA, mutate
from bin_packing import utils as bin_packing_utils
from bin_packing import bin_packing as bin_packing
from engine.algorithm import GeneticSettings, SelectionMethod, run_genetic_algorithm
from sudoku import utils
from sudoku.crossover import cycle_crossover_2d
from sudoku.fitness import calculate_sudoku_fitness
from sudoku.genes import individual_generator
from sudoku.mutations import invert_mutation_sudoku
from sudoku.dataset import games, GAME_SOLUTIONS
from bin_packing.bin_packing import MutationOperators
import concurrent.futures


def main():
    """Main function to run the selected genetic algorithm."""
    display_menu()
    # choice = input("Enter your choice (0, 1, 2, 3, 4, 5, 6, 7, 8): ")
    choice = input("Enter your choice (0, 1, 2, 3, 4): ")

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
        '4': run_bin_packing_with_mutation_function
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