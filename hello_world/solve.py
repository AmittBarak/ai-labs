
def solve():
    solutions = []
    for game in games:
        chosen_game = game
        # Run the genetic algorithm
        best_individual, best_fitness, all_fitness_scores, all_generations = run_genetic_algorithm(
            # 435/498
            GeneticSettings(
                population_size=200,
                genes_count=81,
                elite_size=0.3,
                max_generations=300,
                mutation_rate=0.1,
                selection=SelectionMethod.RANK,
                use_aging=True,
                print_function=utils.print_pretty_grid,
                include_prints=True,
                crossover_generator=cycle_crossover_2d,
                mutation_generator=invert_mutation_generator(chosen_game),
                fitness_calculator=calculate_sudoku_fitness(chosen_game),
                individual_generator=genes_generator(chosen_game),  # todo: runner interface
            ),
        )
        solutions.append(best_individual)
        break
    for solution, game_solution in zip(solutions, games_solutions):
        print(f"Solution:")
        utils.print_pretty_grid_diff(solution, game_solution)
        print(f"Is valid: {utils.is_valid_sudoku(solution)}")