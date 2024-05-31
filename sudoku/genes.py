import random
import typing

from sudoku.utils import print_pretty_grid


def individual_generator(given_grid: list[list[int]]) -> typing.Callable[[int], any]:
    """
    Create an individual Sudoku grid by filling in the empty cells randomly (create the genes).

    Parameters:
    given_grid: The initial Sudoku grid with given numbers (0 for empty cells).

    Returns:
    list: A Sudoku grid with random numbers in the empty cells.
    """

    # def generator(num_genes: int) -> any:
    #     individual = [row[:] for row in given_grid]  # Create a copy of the given grid
    #
    #     for row in range(9):
    #         missing_numbers = [num for num in range(1, 10) if num not in given_grid[row]]
    #         random.shuffle(missing_numbers)
    #         index = 0
    #
    #         for col in range(9):
    #             if individual[row][col] == 0:  # If the cell is empty
    #                 individual[row][col] = missing_numbers[index]
    #                 index += 1
    #
    #     # Flatten the 2D grid to a 1D array
    #     individual = [num for row in individual for num in row]
    #     return individual
    def generator(num_genes: int) -> any:
        individual = [row[:] for row in given_grid]  # Create a copy of the given grid

        def is_valid(num: int, row: int, col: int) -> bool:
            # Check if num is not in the current row, column, and 3x3 subgrid
            if num in individual[row]:
                return False
            if num in (individual[r][col] for r in range(9)):
                return False
            subgrid_row_start = (row // 3) * 3
            subgrid_col_start = (col // 3) * 3
            for r in range(subgrid_row_start, subgrid_row_start + 3):
                for c in range(subgrid_col_start, subgrid_col_start + 3):
                    if individual[r][c] == num:
                        return False
            return True

        def fill_grid() -> bool:
            for row in range(9):
                for col in range(9):
                    if individual[row][col] == 0:  # If the cell is empty
                        random_numbers = list(range(1, 10))
                        random.shuffle(random_numbers)
                        for num in random_numbers:
                            if is_valid(num, row, col):
                                individual[row][col] = num
                                if fill_grid():
                                    return True
                                individual[row][col] = 0
                        return False
            return True

        if not fill_grid():
            raise ValueError("No valid solution exists for the given Sudoku grid")

        # Flatten the 2D grid to a 1D array
        individual_flat = [num for row in individual for num in row]
        return individual_flat

    return generator


if __name__ == '__main__':
    print_pretty_grid(
        individual_generator(
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
