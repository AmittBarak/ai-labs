import random


def sudoku_partially_matched_crossover(parent1, parent2):
    """
    Performs Partially Matched Crossover (PMX) on two parent individuals.
    This function takes two parent individuals and performs Partially Matched Crossover (PMX) to produce two child
    individuals. The PMX algorithm works as follows:
    1. Select index in random between 0 and the length of the parent individual.
    2. Interchange the genes between the two parents at the selected index.
    3. Find the corresponding gene in the other parent and replace the gene in the first parent with the gene in the
         second parent.
    4. Pefrom several iterations of the above steps to produce two child individuals.
    Args:
        parent1: The first parent individual.
        parent2: The second parent individual.
    Returns:
        Two child individuals produced by the PMX algorithm.
    """

    def pmx(parent1, parent2):
        # Initialize the children as copies of the parents
        child1, child2 = parent1.copy(), parent2.copy()

        # Select a random index
        idx1, idx2 = random.sample(range(len(parent1)), 2)
        start_idx, end_idx = min(idx1, idx2), max(idx1, idx2)

        # Copy the selected slice from parent1 to child2 and from parent2 to child1
        for i in range(start_idx, end_idx):
            gene1, gene2 = parent1[i], parent2[i]
            child1[child1.index(gene2)], child1[i] = child1[i], child1[child1.index(gene2)]
            child2[child2.index(gene1)], child2[i] = child2[i], child2[child2.index(gene1)]

        return child1, child2

    return pmx(parent1, parent2)


# #example
# parent1 = [8, 2, 4, 3, 7, 5, 1, 0, 9, 6]
# parent2 = [4, 1, 7, 6, 2, 8, 3, 9, 5, 0]
# child1, child2 = sudoku_partially_matched_crossover(parent1, parent2)
# print('child1:', child1)
# print('child2:', child2)

def cycle_crossover(parent1, parent2):
    child1, child2 = [-1] * len(parent1), [-1] * len(parent2)
    cycle_index = 0
    cycle_count = 0
    visited_indices = set()
    print('parent1:', parent1)
    print('parent2:', parent2)
    while len(visited_indices) < len(parent1):

        if cycle_index in visited_indices:
            cycle_index = next((i for i in range(len(parent1)) if i not in visited_indices), None)
            if cycle_index is None:
                break

        # Find the next cycle in parent1
        cycle = []
        start_idx = cycle_index
        while start_idx not in cycle:
            print(f'start_idx: {start_idx} parent1[start_idx]: {parent1[start_idx]} parent2[start_idx]: {parent2[start_idx]}')
            cycle.append(start_idx)
            start_idx = parent2.index(parent1[start_idx])

            # Copy cycle from parent1 to child1 and from parent2 to child2 if its event index cycle (cycle_0, cycle_2..)
            if cycle_count % 2 == 0:
                for i, idx in enumerate(cycle):
                    child1[idx] = parent1[idx]
                    child2[idx] = parent2[idx]
            else:
                for i, idx in enumerate(cycle):
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]

        # Update visited indices
        visited_indices.update(cycle)
        cycle_count = +1
        cycle_index = next((i for i in range(len(parent1)) if i not in visited_indices), None)

    return child1, child2


def cycle_crossover_2d(parent1, parent2):
    child1, child2 = [], []
    for i in range(len(parent1)):
        child1_row, child2_row = cycle_crossover(parent1[i], parent2[i])
        print('child1_row:', child1_row)
        print('child2_row:', child2_row)
        child1.append(child1_row)
        child2.append(child2_row)
    child1 = [gene for row in child1 for gene in row]
    child2 = [gene for row in child2 for gene in row]
    return child1, child2
