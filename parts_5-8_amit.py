
def selection_pressure_exploitation_factor(population, selection_pressure):
    """
    Calculate the selection pressure exploitation factor.
    """
    return 1 - selection_pressure * (1 - 1 / len(population))


def genetic_diversity_by_genes_distance(population):
    """
    Calculate the genetic diversity of a population by genes distance.
    """
    distance = 0
    for i in range(len(population) - 1):
        distance += sum([1 for a, b in zip(population[i], population[i + 1]) if a != b])
    return distance / len(population[0])


def count_unique_alleles(population):
    """
    Count the number of unique alleles in a population.
    """
    return len({tuple(individual) for individual in population})