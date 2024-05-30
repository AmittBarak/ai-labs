import random
import typing

from sudoku.utils import print_pretty_grid


def genes_generator(given_grid: list[list[int]]) -> typing.Callable[[], list[int]]:
    """
    Create an individual Sudoku grid by filling in the empty cells randomly (create the genes).

    Parameters:
    given_grid: The initial Sudoku grid with given numbers (0 for empty cells).

    Returns:
    list: A Sudoku grid with random numbers in the empty cells.
    """

    def generator() -> list[int]:
        individual = [row[:] for row in given_grid]  # Create a copy of the given grid

        for row in range(9):
            missing_numbers = [num for num in range(1, 10) if num not in given_grid[row]]
            random.shuffle(missing_numbers)
            index = 0

            for col in range(9):
                if individual[row][col] == 0:  # If the cell is empty
                    individual[row][col] = missing_numbers[index]
                    index += 1

        # Flatten the 2D grid to a 1D array
        individual = [num for row in individual for num in row]
        return individual

    return generator


if __name__ == '__main__':
    print_pretty_grid(
        genes_generator(
            [
                [0, 0, 0, 2, 6, 0, 7, 0, 1],
                [6, 8, 0, 0, 7, 0, 0, 9, 0],
                [1, 9, 0, 0, 0, 4, 5, 0, 0],
                [8, 2, 0, 1, 0, 0, 0, 4, 0],
                [0, 0, 4, 6, 0, 2, 9, 0, 0],
                [0, 5, 0, 0, 0, 3, 0, 2, 8],
                [0, 0, 9, 3, 0, 0, 0, 7, 4],
                [0, 4, 0, 0, 5, 0, 0, 3, 6],
                [7, 0, 3, 0, 1, 8, 0, 0, 0]
            ]
        )()
    )
