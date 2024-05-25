

def sudoku_fitness_generator(game_grid):
    def sudoku_fitness(individual):
        """
        Calculates the fitness of a Sudoku individual.
        This function calculates the fitness of a Sudoku individual by checking the rows, columns, and 3x3 subgrids for
        duplicates. The fitness is calculated as the number of unique values in each row, column, and subgrid.
        Args:
            individual: The Sudoku individual to evaluate.
        Returns:
            The fitness score of the Sudoku individual.
        """
        fitness = 0
        # Check the rows
        for row in range(9):
            fitness += len(set(individual[row * 9:(row + 1) * 9]))

        # Check the columns
        for col in range(9):
            fitness += len(set(individual[col::9]))

        # Check the 3x3 squares
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                fitness += len(set(individual[row * 9 + col:row * 9 + col + 3] +
                                   individual[(row + 1) * 9 + col:(row + 1) * 9 + col + 3] +
                                   individual[(row + 2) * 9 + col:(row + 2) * 9 + col + 3]))

        for row in range(9):
            for col in range(9):
                if game_grid[row][col] == individual[row * 9 + col]:
                    fitness += 1

        return fitness

    return sudoku_fitness
