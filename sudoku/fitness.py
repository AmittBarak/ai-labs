import numpy as np

# def calculate_sudoku_fitness(game_grid):
#     def calculate_fitness_reward_penalize(individual):
#         individual = np.array(individual).reshape(9, 9)
#         original = np.array(game_grid).reshape(9, 9)
#         fitness = 0
#         for i in range(9):
#             fitness += len(set(individual[i]))  # Row uniqueness
#             fitness += len(set(individual[:, i]))  # Column uniqueness
#
#         for i in range(3):
#             for j in range(3):
#                 subgrid = individual[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3].flatten()
#                 fitness += len(set(subgrid))  # Subgrid uniqueness
#
#         # Reward no conflicts in original empty spots
#         for i in range(9):
#             for j in range(9):
#                 if original[i][j] == 0:
#                     # Check for duplicates in the row
#                     if individual[i][j] in individual[i, :] and list(individual[i, :]).count(individual[i][j]) > 1:
#                         fitness -= 5
#
#                     # Check for duplicates in the column
#                     if individual[i][j] in individual[:, j] and list(individual[:, j]).count(individual[i][j]) > 1:
#                         fitness -= 5
#
#                     # Check for duplicates in the subgrid
#                     subgrid = individual[i // 3 * 3:i // 3 * 3 + 3, j // 3 * 3:j // 3 * 3 + 3].flatten()
#                     if individual[i][j] in subgrid and list(subgrid).count(individual[i][j]) > 1:
#                         fitness -= 5
#                     else:
#                         fitness += 5
#
#                 elif original[i][j] != individual[i][j]:
#                     fitness -= 5
#         return fitness
#
#     return calculate_fitness_reward_penalize

def calculate_sudoku_fitness(game_grid):
    def calculate_fitness_reward_penalize(individual):
        individual = np.array(individual).reshape(9, 9)
        original = np.array(game_grid).reshape(9, 9)
        fitness = 0

        # Check row and column uniqueness
        for i in range(9):
            fitness += len(set(individual[i]))  # Row uniqueness
            fitness += len(set(individual[:, i]))  # Column uniqueness

        # Check subgrid uniqueness
        for i in range(3):
            for j in range(3):
                subgrid = individual[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3].flatten()
                fitness += len(set(subgrid))  # Subgrid uniqueness

        # Reward and penalize based on original empty spots
        for i in range(9):
            for j in range(9):
                if original[i][j] == 0:
                    # Check for duplicates in the row
                    if list(individual[i, :]).count(individual[i][j]) > 1:
                        fitness -= 10
                    else:
                        fitness += 5

                    # Check for duplicates in the column
                    if list(individual[:, j]).count(individual[i][j]) > 1:
                        fitness -= 10
                    else:
                        fitness += 5

                    # Check for duplicates in the subgrid
                    subgrid = individual[i // 3 * 3:i // 3 * 3 + 3, j // 3 * 3:j // 3 * 3 + 3].flatten()
                    if list(subgrid).count(individual[i][j]) > 1:
                        fitness -= 5
                    else:
                        fitness += 5
                elif original[i][j] != individual[i][j]:
                    fitness -= 5
        return fitness

    return calculate_fitness_reward_penalize


def test_calculate_fitness_reward_penalize():
    # Perfect solution (no conflicts)
    perfect_solution = [
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9
    ]

    # Original grid with empty cells (0s)
    original_grid = [
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9
    ]

    # Partial solution with some conflicts
    partial_solution_with_conflicts = [
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        5, 3, 4, 2, 8, 6, 1, 7, 9  # Conflict in last row (5, 3, 4 are duplicates)
    ]

    perfect_solution_fitness = calculate_sudoku_fitness(original_grid)(perfect_solution)
    print(perfect_solution_fitness)
    assert perfect_solution_fitness > 200, "Test case for perfect solution failed."
    assert calculate_sudoku_fitness(original_grid)([0] * 81) < 50, "Test case for all zeros failed."
    assert calculate_sudoku_fitness(original_grid)(
        partial_solution_with_conflicts) > 100, "Test case for partial solution with few conflicts failed."

    print("All tests passed.")

if __name__ == "__main__":
        test_calculate_fitness_reward_penalize()

