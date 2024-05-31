import os
import random
from hello_world.hello_world import crossover, fitness_GA, mutate
from bin_packing import utils as bin_packing_utils
from bin_packing import bin_packing as bin_packing
from engine.algorithm import GeneticSettings, SelectionMethod, run_genetic_algorithm
from sudoku import utils
from sudoku.crossover import cycle_crossover_2d
from sudoku.fitness import calculate_sudoku_fitness
from sudoku.genes import individual_generator
from sudoku.mutations import invert_mutation_generator
from sudoku.dataset import games, games_solutions


# Main function to run the selected genetic algorithm
def run_selected_genetic_algorithm():
    print("What do you wish to run?")
    print("1. GA with option of SUS + Linear scaling/RWS + Linear scaling /TOURNAMENT/RWS + RANK")
    print("2. Aging")
    print("3. Aging with option of SUS + Linear scaling/RWS + Linear scaling/TOURNAMENT/RWS + RANK")
    print("4. Sudoku solver")
    print("5. Bin packing")
    print("6. Bin packing with First Fit algorithm")
    print("0. Quit")

    choice = input("Enter your choice (0, 1, 2, 3, 4, 5, 6): ")

    if choice == '0':
        quit()

    elif choice == "1":
        best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
            GeneticSettings(
                use_aging=False,
                genes_count=13,
                population_size=100,
                max_generations=100,
                mutation_rate=0.01,
                selection=get_for_selection_method(),
                fitness_calculator=fitness_GA,
                individual_generator=lambda genes: ''.join(chr(random.randint(32, 126)) for _ in range(genes)),
                mutation_generator=mutate,
                crossover_generator=crossover
            )
        )

        print("Best individual:", best_individual)
        print("Best fitness:", best_fitness)
    elif choice == "2":
        best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
            GeneticSettings(
                use_aging=True,
                genes_count=13,
                population_size=100,
                max_generations=100,
                mutation_rate=0.01,
                selection=0,
                fitness_calculator=fitness_GA,
                individual_generator=lambda genes: ''.join(chr(random.randint(32, 126)) for _ in range(genes)),
                mutation_generator=mutate,
                crossover_generator=crossover
            )
        )
        print("Best individual:", best_individual)
        print("Best fitness:", best_fitness)
    elif choice == "3":
        best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
            GeneticSettings(
                use_aging=True,
                genes_count=13,
                population_size=100,
                max_generations=100,
                mutation_rate=0.01,
                selection=get_for_selection_method(),
                fitness_calculator=fitness_GA,
                individual_generator=lambda genes: ''.join(chr(random.randint(32, 126)) for _ in range(genes)),
                mutation_generator=mutate,
                crossover_generator=crossover
            )
        )
        print("Best individual:", best_individual)
        print("Best fitness:", best_fitness)
    elif choice == '4':
        solutions = []
        for game in games:
            chosen_game = game
            # Run the genetic algorithm
            best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
                # 435/498
                GeneticSettings(
                    population_size=300,
                    genes_count=81,
                    elite_size=0.5,
                    max_generations=150,
                    mutation_rate=0.0001,
                    selection=SelectionMethod.NO_SELECTION,
                    use_aging=True,
                    print_function=utils.print_pretty_grid,
                    verbose=True,
                    crossover_generator=cycle_crossover_2d,
                    mutation_generator=invert_mutation_generator(chosen_game),
                    fitness_calculator=calculate_sudoku_fitness(chosen_game),
                    individual_generator=individual_generator(chosen_game),
                ),
            )
            solutions.append(best_individual)
            break
        for solution, game_solution in zip(solutions, games_solutions):
            print(f"Solution:")
            utils.print_pretty_grid_diff(solution, game_solution)
            print(f"Is valid: {utils.is_valid_sudoku(solution)}")

    elif choice == '5':
        print("Do you wish to use adaptive fitness and not the fixed fitness?")
        adaptive = input("'y' for adaptive fitness and 'n' for no adaptive fitness: ").lower() == 'y'
        print("Do you wish to use aging?")
        use_aging = input("'y' for aging and 'n' for no aging: ").lower() == 'y'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'binpack1.txt')
        problems = bin_packing_utils.read_problems_from_file(file_path)
        for problem_id, items in list(problems.items())[:5]:
            bin_capacity = 150
            population_size = 100
            max_generations = 100
            mutation_rate = 0.01
            print(f"Running genetic algorithm for problem: {problem_id}")
            ga_bin_packing = bin_packing.GeneticAlgorithmBinPacking(items, bin_capacity, population_size, max_generations,
                                                        mutation_rate, adaptive, use_aging)
            best_solution, num_bins_used, best_generation, ga_time = ga_bin_packing.run()
            print(
                f"Best solution for {problem_id} uses {num_bins_used} bins and the number of generations it took "
                f"{best_generation}")
    elif choice == '6':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'binpack1.txt')
        bin_packing_utils.run_first_fit(file_path)


def get_for_selection_method() -> SelectionMethod:
    valid_selection = False
    selection = 5
    while not valid_selection:
        print("Which selection would you like?")
        print("1. SUS + Linear scaling")
        print("2. RWS + Linear scaling")
        print("3. TOURNAMENT")
        print("4. RWS + RANK")
        selection = input("Enter your choice (1, 2, 3, 4): ")
        if selection in ["1", "2", "3", "4"]:
            valid_selection = True
        else:
            print("Invalid choice, please select again.")
    return SelectionMethod(int(selection))


if __name__ == "__main__":
    run_selected_genetic_algorithm()
