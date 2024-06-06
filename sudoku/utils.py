from termcolor import colored


def print_pretty_grid_diff(game_grid, game_solution):
    """
    Pretty prints a Sudoku grid with discrepancies marked by an asterisk.
    Also prints the percentage of how close the grid is to the solution.
    """
    game_solution = [int(i) for i in game_solution]
    correct_cells = 0
    total_cells = 81  # 9x9 grid

    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("---------------------")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            if game_grid[i * 9 + j] == game_solution[i * 9 + j]:
                print(game_grid[i * 9 + j], end=" ")
                correct_cells += 1
            else:
                # Print the number with "*" if it doesn't match the solution
                print(colored(str(game_grid[i * 9 + j]) + "*", 'red'), end=" ")
        print()

    # Calculate and print the percentage of correct cells
    percentage_correct = (correct_cells / total_cells) * 100
    print(f"\nPercentage correct: {percentage_correct:.2f}%")


def print_pretty_grid(game_grid):
    """
    Pretty prints a Sudoku grid with discrepancies marked by an asterisk.
    Also prints the percentage of how close the grid is to the solution.
    """
    correct_cells = 0
    total_cells = 81  # 9x9 grid

    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("---------------------")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
                print(game_grid[i * 9 + j], end=" ")
        print()

    # Calculate and print the percentage of correct cells
    percentage_correct = (correct_cells / total_cells) * 100
    print(f"\nPercentage correct: {percentage_correct:.2f}%")


def is_valid_sudoku(array):
    if len(array) != 81:
        return False

    # Convert the array into a 9x9 grid
    grid = [array[i * 9:(i + 1) * 9] for i in range(9)]

    def is_valid_unit(unit):
        unit = [i for i in unit if i != 0]
        return len(unit) == len(set(unit))

    def is_valid_row(grid):
        for row in grid:
            if not is_valid_unit(row):
                return False
        return True

    def is_valid_col(grid):
        for col in zip(*grid):
            if not is_valid_unit(col):
                return False
        return True

    def is_valid_subgrid(grid):
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = [grid[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
                if not is_valid_unit(subgrid):
                    return False
        return True

    return is_valid_row(grid) and is_valid_col(grid) and is_valid_subgrid(grid)


def calculate_correctness(best_individual, game_solution):
    correct_cells = 0
    total_cells = 81  # 9x9 grid
    game_solution = [int(i) for i in game_solution]
    for i in range(9):
        for j in range(9):
            if best_individual[i * 9 + j] == game_solution[i * 9 + j]:
                correct_cells += 1
    return (correct_cells / total_cells) * 100


def sudoku_stop_condition(game_solution):
    def stop_condition(best_individual, population_fitness) -> bool:
        correctness = calculate_correctness(best_individual, game_solution)
        avg_fitness = sum(population_fitness) / len(population_fitness)
        max_fitness = max(population_fitness)

        if correctness == 100:
            print("Solution found!")
            return True
        if avg_fitness == max_fitness:
            print("Population has converged!")
            return True
        return False

    return stop_condition


def adjust_parameters(settings, fitness_variance, top_avg_selection, generation):
    """Adjust parameters based on fitness variance, top average selection, and generation phase."""
    try:
        # Define progress as a fraction of the total generations
        progress = generation / settings.max_generations

        # Adjust mutation rate based on fitness variance and top average selection
        if fitness_variance < settings.fitness_variance_threshold:
            settings.mutation_rate = min(settings.mutation_rate * (1 + settings.adjustment_factor), 0.5)
        else:
            settings.mutation_rate = max(settings.mutation_rate * (1 - settings.adjustment_factor), 0.01)

        # Apply further adjustments based on progress through generations
        if progress < 0.3:  # Early stages
            settings.mutation_rate *= (1 + settings.adjustment_factor * 0.5)  # Aggressively increase mutation rate
        elif progress > 0.7:  # Late stages
            settings.mutation_rate *= (1 - settings.adjustment_factor * 0.5)  # Aggressively reduce mutation rate
        else:  # Middle stages
            settings.mutation_rate *= (1 + settings.adjustment_factor * 0.2)  # Moderate increase in mutation rate

        # Check for stagnation and apply more aggressive adjustments if stuck
        if fitness_variance == 0 and top_avg_selection == 1:
            settings.mutation_rate = min(settings.mutation_rate * 2, 0.5)  # Double the mutation rate

        # Ensure mutation rate stays within bounds
        settings.mutation_rate = max(min(settings.mutation_rate, 0.5), 0.01)
    except Exception as e:
        print(f"Error adjusting parameters: {e}")
        raise





