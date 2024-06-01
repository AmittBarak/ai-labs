import random

import numpy as np


def rws(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current > pick:
            return individual


def rws_linear_scaling(population, fitnesses, a=1.0, b=0.0, min_val=0.0, max_val=1.0):
    fitnesses = np.array(fitnesses)
    min_fitness = fitnesses.min()
    max_fitness = fitnesses.max()

    # Apply linear scaling to fitness values
    scaled_fitnesses = a * (fitnesses - min_fitness) / (max_fitness - min_fitness) + b
    scaled_fitnesses = np.clip(scaled_fitnesses, min_val, max_val)
    scaled_fitnesses = min_val + scaled_fitnesses * (max_val - min_val)

    # Handle the case where all fitness values are the same
    if max_fitness == min_fitness:
        scaled_fitnesses = np.full_like(scaled_fitnesses, min_val + (max_val - min_val) / 2)

    # Compute selection probabilities and select an individual
    selection_probs = scaled_fitnesses / scaled_fitnesses.sum()
    selected_index = np.random.choice(len(population), p=selection_probs)

    return population[selected_index]


def sus_linear_scaling(population, fitnesses, a=1.0, b=0.0, min_val=0.0, max_val=1.0):
    fitnesses = np.array(fitnesses)
    min_fitness = fitnesses.min()
    max_fitness = fitnesses.max()

    # Apply linear scaling to fitness values
    if max_fitness != min_fitness:
        scaled_fitnesses = a * (fitnesses - min_fitness) / (max_fitness - min_fitness) + b
        scaled_fitnesses = min_val + scaled_fitnesses * (max_val - min_val)
    else:
        # Handle the case where all fitness values are the same
        scaled_fitnesses = np.full_like(fitnesses, min_val + (max_val - min_val) / 2)

    # Compute selection probabilities
    total_fitness = np.sum(scaled_fitnesses)
    selection_probs = scaled_fitnesses / total_fitness

    num_selections = len(population)
    start = np.random.uniform(0, 1.0 / num_selections)
    pointers = start + np.arange(num_selections) * 1.0 / num_selections
    cumsum = np.cumsum(selection_probs)

    selected_indices = []
    i = 0
    for p in pointers:
        while p > cumsum[i]:
            i += 1
        selected_indices.append(i)

    # Select individuals based on the indices
    selected_individuals = [population[idx] for idx in selected_indices]

    return selected_individuals


def sus(population, fitnesses):
    total_fitness = sum(fitnesses)
    step = total_fitness / len(population)
    start = random.uniform(0, step)
    points = [start + i * step for i in range(len(population))]
    selected = []
    current_member = 0
    current_fitness = fitnesses[0]
    for point in points:
        while current_fitness < point:
            current_member += 1
            current_fitness += fitnesses[current_member]
        selected.append(population[current_member])
    return random.choice(selected)


def rank(population, fitnesses):
    ranks = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
    rank_probs = [rank / sum(ranks) for rank in ranks]
    return population[np.random.choice(len(population), p=rank_probs)]


def tournament(population, fitnesses, k=3, p=0.7):
    # If we'll make k=1 it's a 1-way tournament which is random so we'll make it a bit less random
    # If we'll make p=1 the selection becomes Deterministic and we don't want that
    selected = random.sample(range(len(population)), k)
    best_idx = max(selected, key=lambda i: fitnesses[i])

    if random.random() < p:
        return population[best_idx]
    else:
        selected.remove(best_idx)
        return population[random.choice(selected)]
