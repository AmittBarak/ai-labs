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


def print_pretty_grid(game_grid: [int]):
    """
    Pretty prints a Sudoku grid.
    """
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("---------------------")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end="")
            print(game_grid[i * 9 + j], end=" ")
        print()

