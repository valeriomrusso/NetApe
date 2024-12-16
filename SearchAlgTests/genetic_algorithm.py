import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import get_valid_moves
from typing import Tuple, List
import random

# PSEUDOCODE
'''
function GENETIC-ALGORITHM(population, fitness) returns an individual
    repeat
        weights <- WEIGHTED-BY(population, fitness)
        population2 <- empty list
        for i = 1 to SIZE(population) do
            parent1, parent2 <- WEIGHTED-RANDOM-CHOICES(population, weights, 2) 
            child <- REPRODUCE(parent1, parent2)
            if (small random probability) then child <- MUTATE(child) 
            add child to population2
        population <- population2
    until some individual is fit enough, or enough time has elapsed 
    return the best individual in population, according to fitness

function REPRODUCE(parent1, parent2) returns an individual
    n <- LENGTH(parent1)
    c <- random number from 1 to n
    return APPEND(SUBSTRING(parent1, 1, c), SUBSTRING(parent2, c + 1, n))


'''

def genetic_algorithm(game_map, start, target, population, max_generations=100, fitness_threshold=0.95):
    """
    Runs a genetic algorithm to optimize the population for pathfinding in NetHack.

    :param population: List of individuals (paths)
    :param fitness: Function that evaluates the fitness of an individual
    :param game_map: 2D numpy array representing the game map
    :param start: Starting position (x, y)
    :param target: Target position (x, y)
    :param max_generations: Maximum number of iterations before stopping
    :param fitness_threshold: Fitness value considered "good enough" to stop the algorithm
    :return: The best path in the population
    """
    generation = 0

    while generation < max_generations:
        # Calculate weights based on fitness scores
        weights = [fitness(individual, game_map, start, target) for individual in population]

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Create a new population
        new_population = []
        for _ in range(len(population)):
            # Select two parents based on their weights
            parent1, parent2 = random.choices(population, weights=weights, k=2)

            # Reproduce to create a child
            child = reproduce(parent1, parent2)

            # Apply mutation with a small probability
            if random.random() < 0.1:  # Mutation probability is 10%
                child = mutate(child, game_map)

            new_population.append(child)

        population = new_population

        # Check if any individual meets the fitness threshold
        best_individual = max(population, key=lambda ind: fitness(ind, game_map, start, target))
        if fitness(best_individual, game_map, start, target) >= fitness_threshold:
            break

        generation += 1

    # Return the best individual based on fitness
    return max(population, key=lambda ind: fitness(ind, game_map, start, target))

def reproduce(parent1, parent2):
    """
    Combines two parents to create a child.

    :param parent1: The first parent (path)
    :param parent2: The second parent (path)
    :return: A new path
    """
    n = min(len(parent1), len(parent2))
    if n <= 1:
        print(f"Warning: Parents too short for crossover: len(parent1)={len(parent1)}, len(parent2)={len(parent2)}")
        return parent1 if random.random() < 0.5 else parent2

    c = random.randint(1, n - 1)  # Crossover point
    return parent1[:c] + parent2[c:]


def mutate(individual, game_map):
    """
    Mutates a path by replacing one of its steps with a random valid move.

    :param individual: The individual (path) to mutate
    :param game_map: 2D numpy array representing the game map
    :return: A mutated path
    """
    if len(individual) > 1:
        mutation_index = random.randint(0, len(individual) - 2)
        current_position = individual[mutation_index]
        valid_moves = get_valid_moves(game_map, current_position)
        if valid_moves:
            next_step = random.choice(valid_moves)
            # Ensure mutation doesn't create invalid diagonal moves
            mutated = individual[:mutation_index + 1] + [next_step] + individual[mutation_index + 2:]
            return mutated
    return individual

def get_random_valid_move(game_map, position):
    """
    Gets a random valid move from a given position.

    :param game_map: 2D numpy array representing the game map
    :param position: Current position (x, y)
    :return: A random valid move (x, y)
    """
    valid_moves = get_valid_moves(game_map, position)
    return random.choice(valid_moves) if valid_moves else position

def get_valid_moves(game_map, position):
    """
    Gets all valid moves from a given position.

    :param game_map: 2D numpy array representing the game map
    :param position: Current position (x, y)
    :return: A list of valid moves [(x1, y1), (x2, y2), ...]
    """
    x, y = position
    moves = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
    return [move for move in moves if is_valid_move(game_map, move)]

def is_valid_move(game_map, position):
    """
    Checks if a move is valid (within bounds and not a wall).

    :param game_map: 2D numpy array or list representing the game map
    :param position: Position to check (x, y)
    :return: True if the move is valid, False otherwise
    """
    if not isinstance(game_map, np.ndarray):
        game_map = np.array(game_map)

    x, y = position
    return 0 <= x < game_map.shape[0] and 0 <= y < game_map.shape[1] and game_map[x, y] != 1

def build_path(parent, target):
    """
    Builds the path from the target to the start using the parent dictionary.

    :param parent: Dictionary mapping each node to its predecessor
    :param target: Target node
    :return: The reconstructed path as a list of nodes
    """
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = parent[current]
    return path[::-1]

def validate_path(path):
    """
    Ensures that a path contains only valid orthogonal moves.

    :param path: List of positions in the path
    :return: True if valid, False otherwise
    """
    for i in range(1, len(path)):
        dx = abs(path[i][0] - path[i - 1][0])
        dy = abs(path[i][1] - path[i - 1][1])
        if dx > 1 or dy > 1 or (dx == 1 and dy == 1):
            return False  # Invalid diagonal or out-of-bounds move
    return True


# ----------------------------------------------

def fitness(individual, game_map, start, target):
    """
    Calculates the fitness of a path based on its length and proximity to the target.

    :param individual: The path to evaluate
    :param game_map: 2D numpy array representing the game map
    :param start: Starting position (x, y)
    :param target: Target position (x, y)
    :return: A fitness score (higher is better)
    """
    if individual[-1] == target:
        return 1.0 / len(individual)  # Reward shorter paths
    return 1.0 / (len(individual) + manhattan_distance(individual[-1], target))

def manhattan_distance(pos1, pos2):
    """
    Calculates the Manhattan distance between two points.

    :param pos1: First point (x, y)
    :param pos2: Second point (x, y)
    :return: Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def generate_initial_population(game_map, start, target, population_size, step_limit):
    """
    Generates an initial population of random paths with a step limit.

    :param game_map: 2D numpy array representing the game map
    :param start: Starting position (x, y)
    :param target: Target position (x, y)
    :param population_size: Number of individuals in the population
    :param step_limit: Maximum number of steps for any path
    :return: A list of individuals (paths)
    """
    population = []

    for _ in range(population_size):
        path = [start]
        current = start
        steps = 0

        while steps < step_limit:
            valid_moves = get_valid_moves(game_map, current)
            if not valid_moves:
                break  # No valid moves, terminate this path
            current = random.choice(valid_moves)
            if current in path:
                break  # Prevent loops
            path.append(current)
            steps += 1

            if current == target:
                break  # Stop if the target is reached

        population.append(path)

    return population
