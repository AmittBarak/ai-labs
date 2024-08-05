import random
from typing import List, Tuple, Dict, Union

class GEPChromosome:
    def __init__(self, length: int):
        self.genes: List[str] = [random.choice(['+', '-', '*', '/', 'x', '1', '2', '3', '4', '5']) for _ in range(length)]
        self.cache: Dict[Tuple, float] = {}

    def evaluate(self, x: float) -> float:
        key = tuple(self.genes) + (x,)
        if key in self.cache:
            return self.cache[key]

        stack: List[float] = []
        for gene in self.genes:
            if gene in {'+', '-', '*', '/'}:
                if len(stack) < 2:
                    return float('inf')  # Invalid expression
                b, a = stack.pop(), stack.pop()
                if gene == '+':
                    stack.append(a + b)
                elif gene == '-':
                    stack.append(a - b)
                elif gene == '*':
                    stack.append(a * b)
                elif gene == '/':
                    stack.append(a / b if b != 0 else 1)  # Assume division by zero results in 1
            elif gene == 'x':
                stack.append(x)
            else:
                stack.append(float(gene))

        result = stack[0] if stack else float('inf')
        self.cache[key] = result
        return result

def fitness_gep(chromosome: GEPChromosome, data: List[Tuple[float, float]]) -> float:
    error = sum((chromosome.evaluate(x) - y) ** 2 for x, y in data)
    bloat_penalty = len(chromosome.genes) * 0.2  # Increased penalty for length
    return 1 / (1 + error + bloat_penalty)  # Higher fitness for lower error and smaller size

def gep_to_expression(chromosome: GEPChromosome) -> str:
    stack: List[str] = []
    for gene in chromosome.genes:
        if gene in {'+', '-', '*', '/'}:
            if len(stack) < 2:
                return "Invalid expression"
            b, a = stack.pop(), stack.pop()
            stack.append(f"({a} {gene} {b})")
        else:
            stack.append(gene)
    return stack[0] if stack else "Invalid expression"

def crossover_gep(parent1: GEPChromosome, parent2: GEPChromosome) -> GEPChromosome:
    child = GEPChromosome(len(parent1.genes))
    crossover_point = random.randint(0, len(parent1.genes) - 1)
    child.genes[:crossover_point] = parent1.genes[:crossover_point]
    child.genes[crossover_point:] = parent2.genes[crossover_point:]
    return child

def mutate_gep(chromosome: GEPChromosome, mutation_rate: float = 0.1) -> None:
    for _ in range(int(mutation_rate * len(chromosome.genes))):
        index = random.randint(0, len(chromosome.genes) - 1)
        chromosome.genes[index] = random.choice(['+', '-', '*', '/', 'x', '1', '2', '3', '4', '5'])

def gep_algorithm(data: List[Tuple[float, float]]) -> GEPChromosome:
    population_size = 1000
    generations = 500
    chromosome_length = 40

    population = [GEPChromosome(chromosome_length) for _ in range(population_size)]

    for generation in range(generations):
        population.sort(key=lambda c: fitness_gep(c, data), reverse=True)

        if fitness_gep(population[0], data) > 0.99:
            return population[0]

        new_population = population[:20]  # Elitism

        while len(new_population) < population_size:
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = crossover_gep(parent1, parent2)

            # Dynamically adjust mutation rate
            mutate_gep(child, mutation_rate=0.1 + (0.9 * (generations - generation) / generations))

            new_population.append(child)

        population = new_population

    return population[0]

def fitness_gep_with_bloat_control(chromosome: GEPChromosome, data: List[Tuple[float, float]]) -> float:
    error = sum((chromosome.evaluate(x) - y) ** 2 for x, y in data)
    bloat_penalty = len(chromosome.genes) * 0.2  # Increased penalty proportional to the length of the chromosome
    return 1 / (1 + error + bloat_penalty)