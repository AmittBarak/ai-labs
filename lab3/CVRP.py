import numpy as np
import os
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple


# Ackley Function
def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    sum2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return a + np.exp(1) + sum1 + sum2


# Ant Colony Optimization
class AntColonyOptimization:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, q, distances, demands, vehicle_capacity):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.distances = distances
        self.num_cities = len(distances)
        self.pheromone = np.ones((self.num_cities, self.num_cities)) / self.num_cities
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity

    def run(self):
        best_routes = None
        best_length = float('inf')

        for _ in range(self.num_iterations):
            all_routes = self.construct_solutions()
            self.update_pheromone(all_routes)
            for routes, route_length in all_routes:
                if route_length < best_length:
                    best_routes = routes
                    best_length = route_length

        return best_routes, best_length

    def construct_solutions(self):
        all_routes = []
        for _ in range(self.num_ants):
            routes, route_length = self.construct_solution()
            all_routes.append((routes, route_length))
        return all_routes

    def construct_solution(self):
        routes = []
        visited = set()
        total_length = 0

        while len(visited) < self.num_cities - 1:
            route = []
            vehicle_load = 0
            current_city = 0
            route.append(current_city)

            while True:
                probabilities = self.transition_probabilities(current_city, visited)
                available_cities = [city for city in range(self.num_cities) if city not in visited and city != 0]
                if np.sum(probabilities) == 0 or len(available_cities) == 0:
                    break
                next_city = self.choose_next_city(probabilities, available_cities)
                if vehicle_load + self.demands[next_city] <= self.vehicle_capacity:
                    vehicle_load += self.demands[next_city]
                    visited.add(next_city)
                    route.append(next_city)
                    current_city = next_city
                else:
                    break

            route.append(0)  # Return to depot
            total_length += self.route_length(route)
            routes.append(route)

        return routes, total_length

    def transition_probabilities(self, current_city, visited):
        probabilities = np.zeros(self.num_cities)
        pheromone = np.copy(self.pheromone[current_city])
        visibility = np.array([1 / (self.distances[current_city][i] + 1e-10) for i in range(self.num_cities)])

        for city in visited:
            pheromone[city] = 0
            visibility[city] = 0

        pheromone = np.power(pheromone, self.alpha)
        visibility = np.power(visibility, self.beta)
        probabilities = pheromone * visibility
        if np.sum(probabilities) > 0:
            probabilities /= np.sum(probabilities)

        return probabilities

    def choose_next_city(self, probabilities, available_cities):
        probabilities = probabilities[available_cities]
        probabilities /= np.sum(probabilities)
        return np.random.choice(available_cities, p=probabilities)

    def route_length(self, route):
        length = 0
        for i in range(len(route) - 1):
            length += self.distances[route[i]][route[i + 1]]
        return length

    def update_pheromone(self, all_routes):
        self.pheromone *= (1 - self.rho)
        for routes, route_length in all_routes:
            for route in routes:
                for i in range(len(route) - 1):
                    self.pheromone[route[i]][route[i + 1]] += self.q / route_length
                    self.pheromone[route[i + 1]][route[i]] += self.q / route_length


# Simulated Annealing
class SimulatedAnnealing:
    def __init__(self, initial_solution: List[List[int]], distances: List[List[float]], initial_temp: float, cooling_rate: float):
        self.solution = initial_solution
        self.distances = distances
        self.temp = initial_temp
        self.cooling_rate = cooling_rate

    def run(self, iterations: int) -> Tuple[List[List[int]], float]:
        current_solution = self.solution
        current_length = self.route_length(current_solution)
        best_solution = current_solution
        best_length = current_length

        for _ in range(iterations):
            new_solution = self.perturb_solution(current_solution)
            new_length = self.route_length(new_solution)
            if new_length < current_length or np.random.rand() < np.exp((current_length - new_length) / self.temp):
                current_solution = new_solution
                current_length = new_length
                if new_length < best_length:
                    best_solution = new_solution
                    best_length = new_length
            self.temp *= self.cooling_rate

        return best_solution, best_length

    def perturb_solution(self, solution: List[List[int]]) -> List[List[int]]:
        new_solution = [route.copy() for route in solution]
        route_idx = np.random.randint(len(new_solution))
        route = new_solution[route_idx]
        if len(route) > 2:
            i, j = np.random.choice(len(route), 2, replace=False)
            route[i], route[j] = route[j], route[i]
        return new_solution

    def route_length(self, solution: List[List[int]]) -> float:
        return sum(
            sum(self.distances[route[i]][route[i + 1]] for i in range(len(route) - 1))
            for route in solution
        )


# Iterative Local Search
class IterativeLocalSearch:
    def __init__(self, initial_solution: List[List[int]], distances: List[List[float]]):
        self.solution = initial_solution
        self.distances = distances

    def run(self, iterations: int) -> Tuple[List[List[int]], float]:
        best_solution = self.solution
        best_length = self.route_length(best_solution)

        for _ in range(iterations):
            new_solution = self.local_search(best_solution)
            new_length = self.route_length(new_solution)
            if new_length < best_length:
                best_solution = new_solution
                best_length = new_length

            new_solution = self.perturb_solution(best_solution)
            new_length = self.route_length(new_solution)
            if new_length < best_length:
                best_solution = new_solution
                best_length = new_length

        return best_solution, best_length

    def local_search(self, solution: List[List[int]]) -> List[List[int]]:
        new_solution = [route.copy() for route in solution]
        for route in new_solution:
            if len(route) > 3:
                for i in range(1, len(route) - 1):
                    for j in range(i + 1, len(route) - 1):  # Protect the last depot
                        new_route = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                        if self.route_length([new_route]) < self.route_length([route]):
                            route[:] = new_route

        # Re-check to ensure all routes start and end with 0
        for route in new_solution:
            if route[0] != 0:
                route.insert(0, 0)
            if route[-1] != 0:
                route.append(0)

        return new_solution

    def perturb_solution(self, solution: List[List[int]]) -> List[List[int]]:
        new_solution = [route.copy() for route in solution]
        route_idx = np.random.randint(len(new_solution))
        route = new_solution[route_idx]

        if len(route) > 3:  # Ensure there are enough elements to swap, excluding the depots
            i, j = np.random.choice(range(1, len(route) - 1), 2, replace=False)
            route[i], route[j] = route[j], route[i]

        # Re-ensure the route starts and ends with 0
        if route[0] != 0:
            route.insert(0, 0)
        if route[-1] != 0:
            route.append(0)

        return new_solution

    def route_length(self, solution: List[List[int]]) -> float:
        return sum(
            self.distances[route[i]][route[i + 1]] for route in solution for i in range(len(route) - 1)
        )


class TabuSearch:
    def __init__(self, initial_solution: Tuple[Tuple[int]], distances: List[List[float]], tabu_tenure: int):
        self.solution = initial_solution
        self.distances = distances
        self.tabu_list = deque([], maxlen=tabu_tenure)
        self.tabu_tenure = tabu_tenure

    def run(self, iterations: int) -> Tuple[Tuple[int], float]:
        best_solution = self.solution
        best_length = self.route_length(best_solution)

        for _ in range(iterations):
            new_solution = self.local_search(best_solution)
            new_length = self.route_length(new_solution)
            if new_length < best_length and tuple(new_solution) not in self.tabu_list:
                best_solution = new_solution
                best_length = new_length

            self.tabu_list.append(tuple(new_solution))

        return best_solution, best_length

    def local_search(self, solution: Tuple[int]) -> Tuple[int]:
        new_solution = [list(route) for route in solution]
        for route in new_solution:
            i, j = self.choose_swap_indices(route)
            route[i:j+1] = reversed(route[i:j+1])
        return tuple(tuple(route) for route in new_solution)

    def choose_swap_indices(self, route: List[int]) -> Tuple[int, int]:
        i = random.randint(1, len(route) - 3)
        j = random.randint(i + 1, len(route) - 2)
        return i, j

    def route_length(self, solution: Tuple[int]) -> float:
        return sum(
            self.distances[route[i]][route[i + 1]]
            for route in solution
            for i in range(len(route) - 1)
        )


class GeneticAlgorithmIslandModel:
    def __init__(self, population_size, num_generations, mutation_rate, crossover_rate, distances, num_islands, migration_interval, migration_size):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.distances = distances
        self.num_cities = len(distances)
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.islands = [self.initialize_population() for _ in range(num_islands)]

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = list(range(1, self.num_cities))
            random.shuffle(individual)
            individual = [0] + individual + [0]
            population.append(individual)
        return population

    def fitness(self, individual):
        return sum(self.distances[individual[i]][individual[i + 1]] for i in range(len(individual) - 1))

    def selection(self, population):
        selected = random.choices(population, k=2, weights=[1/self.fitness(ind) for ind in population])
        return selected

    def crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(1, size - 1), 2))
        child1 = parent1[start:end] + [city for city in parent2 if city not in parent1[start:end]]
        child2 = parent2[start:end] + [city for city in parent1 if city not in parent2[start:end]]

        # Remove duplicate cities and ensure all cities are included
        child1 = [0] + list(dict.fromkeys(child1)) + [0]
        child2 = [0] + list(dict.fromkeys(child2)) + [0]

        return child1, child2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(1, len(individual) - 1), 2)
            individual[i], individual[j] = individual[j], individual[i]

        # Ensure the route starts and ends with 0 and all cities are included once
        individual = [0] + list(dict.fromkeys(individual[1:-1])) + [0]

        return individual

    def run(self):
        for generation in range(self.num_generations):
            for island in self.islands:
                new_population = []
                for _ in range(self.population_size // 2):
                    parent1, parent2 = self.selection(island)
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.extend([self.mutate(child1), self.mutate(child2)])
                island[:] = new_population

            if generation % self.migration_interval == 0:
                self.migrate()

        best_individual = min(min(island, key=self.fitness) for island in self.islands)
        best_length = self.fitness(best_individual)
        return best_individual, best_length

    def migrate(self):
        migrants = [random.sample(island, self.migration_size) for island in self.islands]
        for i in range(self.num_islands):
            self.islands[i].extend(migrants[(i + 1) % self.num_islands])
            self.islands[i] = self.islands[i][self.migration_size:]


class ALNS:
    def __init__(self, initial_solution, distances, demands, vehicle_capacity, num_iterations, destroy_operators, repair_operators):
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.distances = distances
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_iterations = num_iterations
        self.destroy_operators = destroy_operators
        self.repair_operators = repair_operators

    def run(self):
        for iteration in range(self.num_iterations):
            destroy_operator = random.choice(self.destroy_operators)
            repair_operator = random.choice(self.repair_operators)

            destroyed_solution = destroy_operator(self.current_solution)
            new_solution = repair_operator(destroyed_solution, self.demands, self.vehicle_capacity, self.distances)

            if self.evaluate(new_solution) < self.evaluate(self.current_solution):
                self.current_solution = new_solution

            if self.evaluate(new_solution) < self.evaluate(self.best_solution):
                self.best_solution = new_solution

        return self.best_solution

    def evaluate(self, solution):
        return sum(
            self.distances[route[i]][route[i + 1]] for route in solution for i in range(len(route) - 1)
        )


def random_destroy_operator(solution):
    new_solution = [route.copy() for route in solution]
    route = random.choice(new_solution)
    if len(route) > 2:
        index_to_remove = random.randint(1, len(route) - 2)  # Make sure we don't remove the depot (0)
        route.pop(index_to_remove)
    return new_solution


def greedy_repair_operator(solution, demands, vehicle_capacity, distances):
    new_solution = [route.copy() for route in solution]
    for route in new_solution:
        if len(route) == 2:  # Only depot, route is empty
            continue
        load = sum(demands[city] for city in route)
        while load < vehicle_capacity:
            best_city = None
            best_increase = float('inf')
            for city in range(len(demands)):
                if city not in route:
                    increase = distances[route[-2]][city] + distances[city][0] - distances[route[-2]][0]
                    if increase < best_increase:
                        best_city = city
                        best_increase = increase
            if best_city is None:
                break
            if load + demands[best_city] <= vehicle_capacity:
                route.insert(-1, best_city)
                load += demands[best_city]
            else:
                break
    return new_solution


# Function to calculate Euclidean distance
def calculate_distances(coords):
    num_nodes = len(coords)
    distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            distances[i][j] = np.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
    return distances


def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    node_coords_section = False
    demand_section = False
    coords = []
    demands = []
    vehicle_capacity = 0

    for line in lines:
        if line.startswith('CAPACITY'):
            vehicle_capacity = int(line.split()[-1])
        elif line.startswith('NODE_COORD_SECTION'):
            node_coords_section = True
            demand_section = False
        elif line.startswith('DEMAND_SECTION'):
            node_coords_section = False
            demand_section = True
        elif line.startswith('DEPOT_SECTION'):
            break
        elif node_coords_section:
            parts = line.split()
            coords.append((int(parts[1]), int(parts[2])))
        elif demand_section:
            parts = line.split()
            demands.append(int(parts[1]))

    return coords, demands, vehicle_capacity


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'E-n22-k4.txt')
coords, demands, vehicle_capacity = read_data(file_path)
distances = calculate_distances(coords)

# Parameters
num_ants = 4
num_iterations = 100
alpha = 1
beta = 2
rho = 0.5
q = 100

# Run ACO
aco = AntColonyOptimization(num_ants, num_iterations, alpha, beta, rho, q, distances, demands, vehicle_capacity)
aco_solution, aco_length = aco.run()

# Parameters for SA, ILS, and TS
initial_temp = 1000
cooling_rate = 0.95
iterations = 5000
tabu_tenure = 15

# Run SA on ACO solution
sa = SimulatedAnnealing(aco_solution, distances, initial_temp, cooling_rate)
sa_solution, sa_length = sa.run(iterations)

# Run ILS on ACO solution
ils = IterativeLocalSearch(aco_solution, distances)
ils_solution, ils_length = ils.run(iterations)

# Run TS on ACO solution
ts = TabuSearch(aco_solution, distances, tabu_tenure)
ts_solution, ts_length = ts.run(iterations)

# Parameters for GA with Island Model
population_size = 50
num_generations = 200
mutation_rate = 0.01
crossover_rate = 0.7
num_islands = 5
migration_interval = 20
migration_size = 5

# Run GA with Island Model
ga_island_model = GeneticAlgorithmIslandModel(population_size, num_generations, mutation_rate, crossover_rate, distances, num_islands, migration_interval, migration_size)
ga_solution, ga_length = ga_island_model.run()

# Parameters for ALNS
initial_solution = aco_solution
distances = distances
demands = demands
vehicle_capacity = vehicle_capacity
num_iterations = 1000
destroy_operators = [random_destroy_operator]
repair_operators = [greedy_repair_operator]

# Run ALNS
alns = ALNS(initial_solution, distances, demands, vehicle_capacity, num_iterations, destroy_operators, repair_operators)
alns_solution = alns.run()

# Helper function to flatten the list of routes
def flatten_routes(routes):
    return [city for route in routes for city in route]

# Function to plot routes for visualization with algorithm name
def plot_routes(routes, coords, algorithm_name):
    plt.figure(figsize=(10, 8))
    for route in routes:
        x = [coords[city][0] for city in route]
        y = [coords[city][1] for city in route]
        plt.plot(x, y, marker='o', linestyle='-', markersize=5, label=f'Route {routes.index(route) + 1}')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Vehicle Routes using {algorithm_name}')
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to print routes in the desired format with algorithm name
def print_routes(routes, cost, algorithm_name):
    print(f"{algorithm_name} Solution:")
    for i, route in enumerate(routes):
        # Ensure each route starts and ends with 0
        if route[0] != 0 or route[-1] != 0:
            # Check if there are multiple 0s within the route
            if route.count(0) > 1:
                # Remove extra 0s from the route
                route = [city for city in route if city != 0]
                # Add 0 to the start and end
                route = [0] + route + [0]
            else:
                # Just add 0 to the start and end
                if route[0] != 0:
                    route = [0] + route
                if route[-1] != 0:
                    route = route + [0]
        print(f"Route #{i + 1}: {' '.join(map(str, route))}")
    print(f"Cost {cost}")

# Example usage for testing
# Test the algorithms on the Ackley function
print("Testing on Ackley Function:")
aco_ackley = ackley_function(np.array(flatten_routes(aco_solution)))
sa_ackley = ackley_function(np.array(flatten_routes(sa_solution)))
ils_ackley = ackley_function(np.array(flatten_routes(ils_solution)))
ts_ackley = ackley_function(np.array(flatten_routes(ts_solution)))
ga_ackley = ackley_function(np.array(flatten_routes([ga_solution])))
alns_ackley = ackley_function(np.array(flatten_routes(alns_solution)))

print_routes(aco_solution, aco_length, "ACO")
plot_routes(aco_solution, coords, "ACO")
print(f"Ackley Value: {aco_ackley}")

print_routes(sa_solution, sa_length, "SA")
plot_routes(sa_solution, coords, "SA")
print(f"Ackley Value: {sa_ackley}")

print_routes(ils_solution, ils_length, "ILS")
plot_routes(ils_solution, coords, "ILS")
print(f"Ackley Value: {ils_ackley}")

print_routes(ts_solution, ts_length, "TS")
plot_routes(ts_solution, coords, "TS")
print(f"Ackley Value: {ts_ackley}")

print_routes([ga_solution], ga_length, "GA Island Model")
plot_routes([ga_solution], coords, "GA Island Model")
print(f"GA Island Model Ackley Value: {ga_ackley}")

print_routes(alns_solution, sum(aco.route_length(route) for route in alns_solution), "ALNS")
plot_routes(alns_solution, coords, "ALNS")
print(f"Ackley Value: {alns_ackley}")