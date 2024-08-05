import random
from typing import Callable, List, Optional, Tuple

class Node:
    def __init__(self, value: str):
        self.value: str = value
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None

def create_random_tree(depth: int, max_depth: int) -> Node:
    if depth >= max_depth or (depth > 0 and random.random() < 0.5):
        return Node(random.choice(['A', 'B']))

    operator = random.choice(['AND', 'OR', 'NOT'])
    node = Node(operator)

    node.left = create_random_tree(depth + 1, max_depth)
    if operator != 'NOT':
        node.right = create_random_tree(depth + 1, max_depth)

    return node

def print_tree(node: Optional[Node], prefix: str = "", is_left: bool = True) -> None:
    if node is not None:
        print(f"{prefix}{'└── ' if is_left else '┌── '}{node.value}")
        if node.left or node.right:
            new_prefix = f"{prefix}{'    ' if is_left else '│   '}"
            print_tree(node.right, new_prefix, False)
            print_tree(node.left, new_prefix, True)

def tree_to_expression(node: Optional[Node]) -> str:
    if node is None:
        return ""
    if node.value in ['A', 'B']:
        return node.value
    if node.value == 'NOT':
        return f"NOT({tree_to_expression(node.left)})"
    return f"({tree_to_expression(node.left)} {node.value} {tree_to_expression(node.right)})"

def evaluate_tree(node: Optional[Node], a: bool, b: bool, cache: dict) -> bool:
    if node is None:
        return False

    key = (node.value, a, b)
    if key in cache:
        return cache[key]

    result = False
    if node.value == 'A':
        result = a
    elif node.value == 'B':
        result = b
    elif node.value == 'AND':
        result = evaluate_tree(node.left, a, b, cache) and evaluate_tree(node.right, a, b, cache)
    elif node.value == 'OR':
        result = evaluate_tree(node.left, a, b, cache) or evaluate_tree(node.right, a, b, cache)
    elif node.value == 'NOT':
        result = not evaluate_tree(node.left, a, b, cache)

    cache[key] = result
    return result

def fitness_ga(tree: Node, target_func: Callable[[bool, bool], bool]) -> float:
    correct = 0
    total_nodes = count_nodes(tree)
    cache = {}

    for a in [False, True]:
        for b in [False, True]:
            if evaluate_tree(tree, a, b, cache) == target_func(a, b):
                correct += 1

    return correct - (total_nodes * 0.1)

def count_nodes(node: Optional[Node]) -> int:
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)

def crossover_ga(parent1: Node, parent2: Node, max_depth: int) -> Node:
    new_tree = clone_tree(parent1)
    crossover_point = random.choice(get_internal_nodes(new_tree) or [new_tree])
    replacement = random.choice(get_nodes(parent2))

    if calculate_depth(new_tree) <= max_depth:
        if crossover_point.value in ['A', 'B']:
            crossover_point.value = replacement.value
        else:
            crossover_point.left = clone_tree(replacement)
            if crossover_point.value != 'NOT':
                crossover_point.right = clone_tree(replacement)

    return new_tree

def mutate_ga(tree: Node, max_depth: int) -> Node:
    internal_nodes = get_internal_nodes(tree)
    node_to_mutate = random.choice(internal_nodes or get_nodes(tree))

    if node_to_mutate.value in ['A', 'B']:
        node_to_mutate.value = 'A' if node_to_mutate.value == 'B' else 'B'
    else:
        new_operator = random.choice(['AND', 'OR', 'NOT'])
        if new_operator == 'NOT':
            node_to_mutate.right = None
        elif node_to_mutate.value == 'NOT' and new_operator != 'NOT' and calculate_depth(tree) <= max_depth:
            node_to_mutate.right = create_random_tree(0, 2)
        node_to_mutate.value = new_operator
    return tree

def clone_tree(node: Optional[Node]) -> Optional[Node]:
    if node is None:
        return None
    new_node = Node(node.value)
    new_node.left = clone_tree(node.left)
    new_node.right = clone_tree(node.right)
    return new_node

def get_nodes(node: Optional[Node]) -> List[Node]:
    if node is None:
        return []
    return [node] + get_nodes(node.left) + get_nodes(node.right)

def get_internal_nodes(node: Optional[Node]) -> List[Node]:
    if node is None or (node.left is None and node.right is None):
        return []
    return [node] + get_internal_nodes(node.left) + get_internal_nodes(node.right)

def calculate_depth(node: Optional[Node]) -> int:
    if node is None:
        return 0
    return 1 + max(calculate_depth(node.left), calculate_depth(node.right))

def genetic_algorithm(target_func: Callable[[bool, bool], bool]) -> Node:
    population_size = 300
    generations = 300
    max_depth = 10

    population = [create_random_tree(0, max_depth) for _ in range(population_size)]

    for _ in range(generations):
        population.sort(key=lambda tree: fitness_ga(tree, target_func), reverse=True)

        if fitness_ga(population[0], target_func) >= 3.9:
            return population[0]

        new_population = population[:10]  # Elitism

        while len(new_population) < population_size:
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = crossover_ga(parent1, parent2, max_depth)

            if random.random() < 0.1:
                child = mutate_ga(child, max_depth)

            new_population.append(child)

        population = new_population

    return population[0]