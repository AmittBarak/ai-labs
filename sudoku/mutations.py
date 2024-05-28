import random


def scramble_mutation(individual: [int]):
    """
    Perform Scramble Mutation on an individual.
    This function takes an individual and performs Scramble Mutation on it. Scramble Mutation works as follows:
    1. Select two positions in the individual at random.
    2. Scramble the values between the two positions.
    Args:
        individual: The individual to mutate.
    Returns:
        The mutated individual.
    """
    individual = list(individual)
    start, end = sorted(random.sample(range(len(individual)), 2))
    random.shuffle(individual[start:end + 1])
    # to 1d
    individual = [num for row in individual for num in row]
    return individual


def swap_mutation(individual):
    """
    Performs Swap Mutation on an individual.
    This function takes an individual and performs Swap Mutation on it. Swap Mutation works as follows:
    1. Select two positions in the individual at random.
    2. Swap the values at the two positions.
    Args:
        individual: The individual to mutate.
    Returns:
        The mutated individual.
    """
    individual = list(individual)
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return ''.join(individual)


def invert_mutation_generator(game_grid):
    def invert_mutation(individual):
        """
        Performs Inversion Mutation on an individual for each row in a Sudoku grid.
        Use the original game grid to determine the rows in the individual.
        Do not change the original game grid.

        This function takes an individual and performs Inversion Mutation on it for each row in a Sudoku grid. Inversion
        Mutation works as follows:
        1. Select two NON-ORIGINAL grid positions in the individual at random.
        2. Invert the values between the two positions.
        Args:
            individual: The individual to mutate.
        Returns:
            The mutated individual.
        """
        individual = list(individual)
        individual_2d = [individual[i * 9:(i + 1) * 9] for i in range(9)]
        for row in range(9):
            original_row = game_grid[row]
            non_original_indices = [i for i in range(9) if original_row[i] == 0]
            if non_original_indices:
                start, end = sorted(random.sample(non_original_indices, 2))
                individual_2d[row][start:end + 1] = individual_2d[row][start:end + 1][::-1]
        individual = [num for row in individual_2d for num in row]
        return individual

    return invert_mutation
