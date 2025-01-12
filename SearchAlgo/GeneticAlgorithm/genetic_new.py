import numpy as np
import random
from utils import get_valid_moves
from typing import Tuple, List
import time
from typing import Dict, Any

def run_multiple_evaluations(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], 
                           population_size: int, generations: int, mutation_rate: float, 
                           max_steps: int, num_iterations: int) -> Dict[str, Any]:
    successful_runs = []
    all_metrics = []
    total_actual_runs = 0  # Counter per il numero effettivo di runs
    
    for i in range(num_iterations):
        paths, metrics = evaluate_genetic_algorithm(game_map, start, target, 
                                                 population_size, generations, 
                                                 mutation_rate, max_steps)
        all_metrics.append(metrics)
        total_actual_runs += 1  # Incrementa il contatore per ogni run effettiva
        if metrics['path_found']:
            successful_runs.append(metrics)
    
    # Calcola il success rate basato sul numero effettivo di runs
    success_rate = (len(successful_runs) / total_actual_runs) * 100
    
    # Calculate averages only for successful runs
    if successful_runs:
        avg_metrics = {
            'avg_execution_time': sum(run['execution_time'] for run in successful_runs) / len(successful_runs),
            'avg_final_path_length': sum(run['final_path_length'] for run in successful_runs) / len(successful_runs),
            'avg_starting_paths': sum(run['starting_paths'] for run in successful_runs) / len(successful_runs),
            'avg_final_paths': sum(run['final_paths'] for run in successful_runs) / len(successful_runs),
            'avg_duplicate_paths': sum(run['number_of_dublicate_best_paths'] for run in successful_runs) / len(successful_runs),
            'avg_generations_needed': sum(run['generations_needed'] for run in successful_runs) / len(successful_runs),
            'manhattan_distance': all_metrics[0]['manhattan_distance'],  # This is constant for all runs
            'success_rate': success_rate,
            'total_runs': num_iterations,
            'successful_runs': len(successful_runs)
        }
    else:
        avg_metrics = {
            'success_rate': 0,
            'total_runs': num_iterations,
            'successful_runs': 0,
            'manhattan_distance': all_metrics[0]['manhattan_distance']
        }
    
    return avg_metrics

def evaluate_genetic_algorithm(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], 
                             population_size: int, generations: int, mutation_rate: float, 
                             max_steps: int) -> Dict[str, Any]:
    start_time = time.time()
    
    # Run genetic algorithm
    paths, final_generation = genetic_alg_func(game_map, start, target, population_size, generations, mutation_rate, max_steps)
    
    end_time = time.time()

    # Split and clean paths
    cleaned_paths = []
    for path in paths:
        if path not in cleaned_paths:
            cleaned_paths.append(path)
    
    # Calculate metrics
    metrics = {
        'execution_time': end_time - start_time,
        'final_path_length': len(cleaned_paths[-1]) if cleaned_paths else 0,
        'starting_paths': len(paths),
        'final_paths': len(cleaned_paths),
        'number_of_dublicate_best_paths': len(paths) - len(cleaned_paths),
        'path_found': bool(paths and paths[-1][-1] == target),
        'generations_needed': final_generation,
        'manhattan_distance': abs(target[0] - start[0]) + abs(target[1] - start[1])
    }
    return cleaned_paths, metrics

# Side functions

def fitness(path: List[Tuple[int, int]], target: Tuple[int, int], population=None) -> float:
    """
    Enhanced fitness function with stronger local maxima avoidance
    """
    if not path:
        return float('-inf')
    
    last_node = path[-1]
    dist = abs(last_node[0] - target[0]) + abs(last_node[1] - target[1])
    
    # Track position history and frequencies
    position_counts = {}
    for pos in path:
        position_counts[pos] = position_counts.get(pos, 0) + 1
    
    # === Local Maxima Detection ===
    local_maxima_penalty = 0
    
    # Check for path stagnation
    last_positions = path[-10:] if len(path) >= 10 else path  # Increased window to 10
    unique_last_positions = len(set(last_positions))
    area_size = max(abs(p[0] - q[0]) + abs(p[1] - q[1]) 
                   for p in last_positions for q in last_positions)
    
    # Penalize small area exploration heavily
    if area_size < 5:  # If exploring area smaller than 5x5
        local_maxima_penalty += 200 * (5 - area_size)  # Increased penalty
    
    # Penalize repeated end positions exponentially
    end_point_count = position_counts.get(last_node, 0)
    if end_point_count > 1:
        local_maxima_penalty += 100 * (2 ** end_point_count)
    
    # === Population Diversity Analysis ===
    diversity_score = 0
    if population:
        similar_paths = 0
        similar_endings = 0
        
        for other_path in population:
            if not other_path:
                continue
                
            # Check for similar endings
            if other_path[-1] == last_node:
                similar_endings += 1
            
            # Check for path similarity
            common_positions = len(set(path).intersection(set(other_path)))
            path_similarity = common_positions / len(set(path))
            
            if path_similarity > 0.7:  # If paths are more than 70% similar
                similar_paths += 1
        
        # Heavy penalties for population convergence
        if similar_paths > len(population) * 0.3:  # If more than 30% paths are similar
            local_maxima_penalty += 300
        
        if similar_endings > len(population) * 0.2:  # If more than 20% end in same spot
            local_maxima_penalty += 500
    
    # === Progress Assessment ===
    target_distances = [abs(pos[0] - target[0]) + abs(pos[1] - target[1]) 
                       for pos in path]
    
    # Calculate progress metrics
    min_dist = min(target_distances)
    progress_variance = np.std(target_distances) if len(target_distances) > 1 else 0
    
    # Reward for getting closer to target at any point
    progress_reward = 50 * (1 / (min_dist + 1))
    
    # Reward for consistent exploration (high variance in distances)
    exploration_reward = progress_variance * 10
    
    # === Path Quality Assessment ===
    repetition_penalty = sum(count * count for count in position_counts.values())
    unique_positions = len(set(path))
    coverage_score = (unique_positions / len(path)) * 100
    
    # === Final Score Calculation ===
    final_score = (
        -(dist * 3)  # Increased base distance penalty
        - local_maxima_penalty
        - (repetition_penalty * 2)
        + coverage_score
        + progress_reward
        + exploration_reward
    )
    
    # Extreme penalty for very poor paths
    if local_maxima_penalty > 1000 or repetition_penalty > 50:
        final_score *= 2  # Double the negative score
    
    return final_score

def generate_random_path(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], max_steps: int) -> List[Tuple[int, int]]:
    """
    Generates a random path starting from the initial node.
    The path develops by choosing valid moves until the target is reached or the maximum step limit is reached.
    """
    path = [start]  # Start from the starting node
    current = start
    for _ in range(max_steps):  # Limit the length of the paths
        neighbors = get_valid_moves(game_map, current)  # Get valid moves
        if not neighbors:  # If there are no valid moves, end the path
            break
        current = random.choice(neighbors)  # Choose a random move
        path.append(current)
        if current == target:  # If it reaches the target, stop
            break
    return path

def is_valid_path(path: List[Tuple[int, int]]) -> bool:
    """
    Checks that the path is valid by ensuring that each step is adjacent to the previous one.
    A step is valid if it moves by only one unit horizontally or vertically.
    """
    if not path or len(path) < 2:
        return True
        
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        
        # Check that the movement is by only one unit horizontally or vertically
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        
        # A step is valid if:
        # - it moves by 1 horizontally and 0 vertically
        # - it moves by 0 horizontally and 1 vertically
        if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
            return False
            
    return True

def crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Combines two parent paths to create a new child path,
    ensuring that the split point produces a valid path.
    """
    if len(parent1) < 3 or len(parent2) < 3:
        return random.choice([parent1, parent2])
    
    max_split = min(len(parent1), len(parent2)) - 1
    for _ in range(max_split):  # Try multiple times to find a valid split
        split = random.randint(1, max_split)  # Choose a random split point
        
        # Check if the connection is valid (movement by one unit vertically or horizontally)
        if (
            (abs(parent1[split - 1][0] - parent2[split][0]) == 1 and parent1[split - 1][1] == parent2[split][1]) or
            (parent1[split - 1][0] == parent2[split][0] and abs(parent1[split - 1][1] - parent2[split][1]) == 1)
        ) and parent1[split - 1] != parent2[split]:
            return parent1[:split] + parent2[split:]
    
    # If no valid split is found, return one of the parents
    return random.choice([parent1, parent2])

def mutate(path: List[Tuple[int, int]], game_map: np.ndarray, mutation_rate: float) -> List[Tuple[int, int]]:
    """
    Applies a significant mutation with probability mutation_rate
    """
    if not path or random.random() < mutation_rate:  # Do not mutate if it does not pass the probabilistic check
        return path
    
    # If we pass the probability check, make a significant mutation
    idx = random.randint(0, len(path) - 1)
    current = path[idx]
    new_subpath = [current]
    
    # Generate some random valid steps
    for _ in range(min(5, len(path) - idx)):
        neighbors = get_valid_moves(game_map, new_subpath[-1])
        if not neighbors:
            break
        new_subpath.append(random.choice(neighbors))
    
    return path[:idx] + new_subpath


# Main function
def genetic_alg_func(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], 
                    population_size: int, generations: int, mutation_rate: float, max_steps: int) -> List[List[Tuple[int, int]]]:
    
    # Initial population
    population = [generate_random_path(game_map, start, target, max_steps) for _ in range(population_size)]
    list_paths = []  # List to save the paths of each generation
    
    for generation in range(generations):
        # Evaluation
        population.sort(key=lambda path: fitness(path, target), reverse=True)
        
        # Save the best path of the generation
        best_path = population[0]
        if generation > 0:
            list_paths.append(best_path)
        
        # Check if we have reached the target
        if best_path and best_path[-1] == target:
            print(f"Target reached in generation {generation}")
            return list_paths, generation
            
        new_population = []
        
        # Reproduction to complete the population
        while len(new_population) < population_size:
            # Select two different parents
            parent1, parent2 = random.sample(population[:population_size//2], 2)
            # Create a child through crossover
            child = crossover(parent1, parent2)
            # Apply mutation
            child = mutate(child, game_map, mutation_rate)
            # Ensure the child is different from the parents
            if child != parent1 and child != parent2:
                new_population.append(child)
        
        # Update the population
        population = new_population[:population_size]  # Ensure correct size
    
    print("Target not reached after all generations.")
    return list_paths, generation
