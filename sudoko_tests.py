import matplotlib.pyplot as plt
from engine.algorithm import GeneticSettings, SelectionMethod, run_genetic_algorithm
from sudoku import utils
from sudoku.crossover import cycle_crossover_2d
from sudoku.fitness import calculate_sudoku_fitness
from sudoku.genes import individual_generator
from sudoku.mutations import invert_mutation_generator
from sudoku.dataset import games, games_solutions

# List of different population sizes to test
population_sizes = [150, 200, 300, 400, 500]
mute_rate = [0.01, 0.02, 0.03, 0.04, 0.05]
max_generations = [100, 150, 200, 250, 300]
use_aging = [True, False]

results = []
param_name = "Use Aging"
params = use_aging
# Function to run the genetic algorithm with a given population size
def run_test(param):
    solutions = []
    for game in games[0:3]:  # Adjust the range as needed
        chosen_game = game
        best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
            GeneticSettings(
                population_size=400,
                genes_count=81,
                elite_size=0.5,
                max_generations=200,
                mutation_rate=0.04,
                selection=SelectionMethod.NO_SELECTION,
                use_aging=param,
                print_function=utils.print_pretty_grid,
                verbose=True,
                crossover_generator=cycle_crossover_2d,
                mutation_generator=invert_mutation_generator(chosen_game),
                fitness_calculator=calculate_sudoku_fitness(chosen_game),
                individual_generator=individual_generator(chosen_game),
            ),
        )
        solutions.append(best_individual)
        break  # Assuming you only want to test with one game at a time

    correct_cells = 0
    total_cells = 81  # 9x9 grid

    for solution, game_solution in zip(solutions, games_solutions):
        game_solution = [int(i) for i in game_solution]
        for i in range(9):
            for j in range(9):
                if solution[i * 9 + j] == game_solution[i * 9 + j]:
                    correct_cells += 1

    percentage_correct = (correct_cells / total_cells) * 100
    return percentage_correct

# Run tests for each population size
for m in params:
    percentage_correct = run_test(m)
    results.append((m, percentage_correct))
    print(f"{param_name}: {m}, Percentage correct: {percentage_correct:.2f}%")

# Plot the results
params_result, percentages_correct = zip(*results)
plt.figure(figsize=(10, 6))
plt.plot(params_result, percentages_correct, marker='o')
plt.title(f'Percentage Correct vs. {param_name}')
plt.xlabel(f'{param_name}')
plt.ylabel('Percentage Correct')
plt.grid(True)
plt.show()