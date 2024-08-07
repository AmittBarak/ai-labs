import random


def to_2d(individual, game_grid):
    individual = list(individual)
    individual_2d = [individual[i * 9:(i + 1) * 9] for i in range(9)]
    original_grid = [int(i) for i in game_grid]
    original_2d = [original_grid[i * 9:(i + 1) * 9] for i in range(9)]
    return individual_2d, original_2d


def scramble_mutation_sudoku(game_grid):
    def scramble_mutation(individual: [int], mutation_rate: float):
        """
        Perform Scramble Mutation on an individual.
        This function takes an individual and performs Scramble Mutation on it.
        The mutation will work only on the non-original cells,
        without changing the original game grid.
        Scramble Mutation works as follows:
        1. Select two positions in the individual at random (from non-original cells) - do it with while.
        2. Scramble the values between the two positions.
        Args:
            individual: The individual to mutate.
        Returns:
            The mutated individual.
        """
        individual = list(individual)
        individual_2d = [individual[i * 9:(i + 1) * 9] for i in range(9)]
        for row in range(9):
            if not random.random() < mutation_rate:
                continue

            original_row = game_grid[row]
            non_original_indices = [i for i in range(9) if original_row[i] == 0]
            if len(non_original_indices) > 1:
                start, end = sorted(random.sample(non_original_indices, 2))
                random.shuffle(individual_2d[row][start:end + 1])

        individual = [num for row in individual_2d for num in row]
        return individual

    return scramble_mutation


def swap_mutation_sudoku(game_grid):
    def swap_mutation(individual: [int], mutation_rate: float):
        """
        Perform Swap Mutation on an individual without changing the original game grid cells (only the non-original cells).
        This function takes an individual and performs Swap Mutation on it. Swap Mutation works as follows:
        1. Select two positions in the individual at random (from non-original cells) - do it with while.
        2. Swap the values at the two positions.
        Args:
            individual: The individual to mutate.
        Returns:
            The mutated individual.
        """
        individual = list(individual)
        individual_2d = [individual[i * 9:(i + 1) * 9] for i in range(9)]
        for row in range(9):
            if not random.random() < mutation_rate:
                continue

            original_row = game_grid[row]
            non_original_indices = [i for i in range(9) if original_row[i] == 0]
            if len(non_original_indices) > 1:
                start, end = sorted(random.sample(non_original_indices, 2))
                individual_2d[row][start], individual_2d[row][end] = individual_2d[row][end], individual_2d[row][start]

        individual = [num for row in individual_2d for num in row]
        return individual

    return swap_mutation


def invert_mutation_sudoku(game_grid):
    def invert_mutation(individual, mutation_rate: float):
        """
        Performs Inversion Mutation on an individual for each row in a Sudoku grid.
        Use the original game grid to determine the rows in the individual.
        Do not change the original game grid.

        This function takes an individual and performs Inversion Mutation on it for each row in a Sudoku grid.
        Inversion
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
            if random.random() < mutation_rate:
                original_row = game_grid[row]
                non_original_indices = [i for i in range(9) if original_row[i] == 0]
                if len(non_original_indices) > 1:  # Ensure there are at least two positions to invert
                    start, end = sorted(random.sample(non_original_indices, 2))
                    individual_2d[row][start:end + 1] = individual_2d[row][start:end + 1][::-1]

        individual = [num for row in individual_2d for num in row]
        return individual

    return invert_mutation



def calculate_accuracy(original: [int], mutated: [int]) -> float:
    """
    Calculate the accuracy of the mutation.
    Accuracy is defined as the percentage of elements in the mutated individual that match the original individual.
    Args:
        original: The original individual.
        mutated: The mutated individual.
    Returns:
        The accuracy percentage.
    """
    matches = sum(1 for o, m in zip(original, mutated) if o == m)
    return (matches / len(original)) * 100


def test_mutations():
    # Test Scramble Mutation
    individual = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    mutated = scramble_mutation(individual)
    accuracy = calculate_accuracy(individual, mutated)
    print(f"Scramble Mutation passed with accuracy: {accuracy:.2f}%")

    # Test Swap Mutation
    individual = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    mutated = swap_mutation(individual)
    accuracy = calculate_accuracy(individual, mutated)
    print(f"Swap Mutation passed with accuracy: {accuracy:.2f}%")

    # Test Invert Mutation
    game_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    individual = list(range(1, 82))
    invert_mutation = invert_mutation_sudoku(game_grid)
    mutated = invert_mutation(individual, 0.1)
    accuracy = calculate_accuracy(individual, mutated)
    print(f"Invert Mutation passed with accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test_mutations()
