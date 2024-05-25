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


def invert_mutation(individual):
    """
    Performs Inversion Mutation on an individual.
    This function takes an individual and performs Inversion Mutation on it. Inversion Mutation works as follows:
    1. Select two positions in the individual at random.
    2. Reverse the order of the values between the two positions (inclusive).
    Args:
        individual: The individual to mutate.
    Returns:
        The mutated individual.
    """
    individual = list(individual)
    start, end = sorted(random.sample(range(len(individual)), 2))
    individual[start:end + 1] = reversed(individual[start:end + 1])
    return ''.join(individual)
