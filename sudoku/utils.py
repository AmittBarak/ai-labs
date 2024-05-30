from termcolor import colored


def aging(population_fitness, age_distribution):
    """
    This function takes a population's population_fitness and age distribution, and
    applies a scaling factor based on age. The scaling factors are:
    - 0.5 for ages below 0.25 (young)
    - 1.5 for ages between 0.25 and 0.75 (adult)
    - 0.5 for ages above 0.75 (old)
    """
    young_scale, adult_scale, old_scale = 0.5, 1.5, 0.5
    thresholds = [0.25, 0.75]
    scales = [young_scale, adult_scale, old_scale]

    return [fitness * scales[sum(age > threshold for threshold in thresholds)]
            for age, fitness in zip(age_distribution, population_fitness)]


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
    grid = [array[i*9:(i+1)*9] for i in range(9)]

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
                subgrid = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
                if not is_valid_unit(subgrid):
                    return False
        return True

    return is_valid_row(grid) and is_valid_col(grid) and is_valid_subgrid(grid)
