from engine.algorithm import GeneticEngine, GeneticConfig, SelectionMethod
from engine.utils import pretty_grid
from sudoku.crossover_functions import cycle_crossover, cycle_crossover_2d
from sudoku.fitness import sudoku_fitness_generator
from sudoku.genes import genes_generator
from sudoku.mutations import scramble_mutation
from sudoku.solver import games


class SudokuGeneticEngine(GeneticEngine):
    def __init__(self, config: GeneticConfig, game_grid: [[int]], mutation_generator: callable,
                 crossover_generator: callable):
        super().__init__(config)
        self.game_grid = game_grid
        self.mutation_generator = mutation_generator
        self.crossover_generator = crossover_generator

    def calculate_fitness(self, individual):
        return sudoku_fitness_generator(self.game_grid)(individual)

    def individual_generator(self):
        return genes_generator(self.game_grid)

    def crossover(self, parent1, parent2):
        return self.crossover_generator(parent1, parent2)

    def mutate(self, individual):
        return self.mutation_generator(individual)


sudoku_solver = SudokuGeneticEngine(
    GeneticConfig(
        population_size=100,
        genes_count=81,
        max_generations=100,
        mutation_rate=0.01,
        selection=SelectionMethod.SUS,
        use_aging=True
    ),
    game_grid=games[0],
    crossover_generator=cycle_crossover_2d,
    mutation_generator=scramble_mutation
)

best_individual, best_fitness, all_fitness_scores, all_generations = sudoku_solver.run_genetic_algorithm()
pretty_grid(best_individual)