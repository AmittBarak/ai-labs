from engine.algorithm import GeneticSettings, SelectionMethod, run_genetic_algorithm
from sudoku.utils import print_pretty_grid
from sudoku.crossover import cycle_crossover_2d
from sudoku.fitness import calculate_sudoku_fitness
from sudoku.genes import genes_generator
from sudoku.mutations import scramble_mutation
from sudoku.dataset import games


def solve():
    solutions = []
    for game in games:
        chosen_game = game
        # Run the genetic algorithm
        best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
            # 435/498
            GeneticSettings(
                population_size=300,
                genes_count=81,
                elite_size=0.3,
                max_generations=200,
                mutation_rate=0.1,
                selection=SelectionMethod.RANK,
                use_aging=False,
                crossover_generator=cycle_crossover_2d,
                mutation_generator=scramble_mutation,
                fitness_calculator=calculate_sudoku_fitness(chosen_game),
                individual_generator=genes_generator(chosen_game),  # todo: runner interface
            ),
        )
        solutions.append(best_individual)
    for solution, i in zip(solutions, range(1, len(solutions) + 1)):
        print(f"Solution {i}")
        print_pretty_grid(solution)
