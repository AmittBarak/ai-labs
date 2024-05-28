def sudoku_fitness_generator(game_grid):
    flattened_game_grid = [num for row in game_grid for num in row]

    def sudoku_fitness(individual):
        """
        Calculates the fitness of a Sudoku individual.

        This function calculates the fitness of a Sudoku individual by checking the columns,
        and 3x3 subgrids for duplicates. The fitness is calculated by subtracting points for repeated numbers.

        Args:
            game_grid (list): The original Sudoku grid with given numbers (0 for empty cells), as a 2D list (list of lists).
            individual (list): The Sudoku individual to evaluate, as a 1D list of 81 elements.

        Returns:
            int: The fitness score of the Sudoku individual.
        """
        fitness = 0

        # Convert the individual from 1D to 2D
        individual_2d = [individual[i * 9:(i + 1) * 9] for i in range(9)]

        # Check columns
        for i in range(9):
            seen = {}
            for j in range(9):  # Check each cell in the column
                if individual_2d[j][i] in seen:
                    seen[individual_2d[j][i]] += 1
                else:
                    seen[individual_2d[j][i]] = 1
            for key in seen:  # Subtract fitness for repeated numbers
                fitness -= (seen[key] - 1)

        # Check 3x3 subgrids
        for m in range(3):  # For each 3x3 square
            for n in range(3):
                seen = {}
                for i in range(3 * m, 3 * (m + 1)):  # Check cells in 3x3 square
                    for j in range(3 * n, 3 * (n + 1)):
                        if individual_2d[i][j] in seen:
                            seen[individual_2d[i][j]] += 1
                        else:
                            seen[individual_2d[i][j]] = 1
                for key in seen:  # Subtract fitness for repeated numbers
                    fitness -= (seen[key] - 1)

        return fitness

    return sudoku_fitness
