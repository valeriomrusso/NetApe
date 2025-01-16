import numpy as np
import random
from utils import get_valid_moves
from typing import Tuple, List, Dict

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state # current state represented by the node
        self.parent = parent # reference to the parent node
        self.children = [] # list of child nodes 
        self.visits = 0 # number of times this node has been visited
        self.value = 0 # cumulative reward associated with this node
        self.path_to_leaf = [] if parent is None else parent.path_to_leaf + [state]
        # Track minimum distance achieved from this node to target
        self.min_distance_to_target = float('inf')
        
    def is_fully_expanded(self, game_map: np.ndarray, cached_moves: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> bool:
        if self.state not in cached_moves:
            cached_moves[self.state] = get_valid_moves(game_map, self.state)
        return len(self.children) == len(cached_moves[self.state])

    def best_child(self, exploration_weight: float = 1.0, target: Tuple[int, int] = None):
        if not self.children:
            return None
            
        scores = []
        for child in self.children:
            # Balance between:
            # 1. UCB1 score
            # 2. Distance progress to target
            # 3. Path length penalty
            exploitation = child.value / (child.visits + 1e-6)
            exploration = exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
            
            if target:
                # Reward progress towards target
                current_distance = manhattan_distance(child.state, target)
                distance_score = 1.0 / (current_distance + 1)
                # Penalize longer paths
                path_length_penalty = 0.1 * len(child.path_to_leaf)
                
                total_score = exploitation + exploration + 0.5 * distance_score - path_length_penalty
            else:
                total_score = exploitation + exploration
                
            scores.append(total_score)
            
        return self.children[np.argmax(scores)]

    def add_child(self, child_state, target: Tuple[int, int]):
        child_node = MCTSNode(child_state, parent=self)
        child_node.min_distance_to_target = manhattan_distance(child_state, target)
        self.children.append(child_node)
        return child_node

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def dynamic_reward(path_length: int, current_distance: int, initial_distance: int, 
                          best_distance: float, visited_count: int) -> float:
    """
    Enhanced reward function that heavily favors shorter paths and direct progress
    """
    if current_distance == 0:  # Target reached
        # Base reward for reaching target, with bonus for shorter paths
        path_efficiency = max(0, 1 - (path_length / (initial_distance * 2)))
        return 1.0 + path_efficiency
        
    # Calculate progress towards target
    progress = (initial_distance - current_distance) / initial_distance
    
    # Penalties
    length_penalty = 0.2 * (path_length / initial_distance)  # Penalty for long paths
    revisit_penalty = 0.1 * visited_count  # Penalty for revisiting states
    
    return max(0.1, progress - length_penalty - revisit_penalty)

def mcts(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], 
         iterations: int = 1000, exploration_factor: float = 0.2) -> List[Tuple[int, int]]:
    
    root = MCTSNode(start)
    best_path = []
    best_distance = float('inf')
    cached_moves = {}
    
    # Initialize distance-based parameters
    initial_distance = manhattan_distance(start, target)
    max_reasonable_path_length = initial_distance * 2  # Reasonable upper bound for path length
    
    for iteration in range(iterations):
        node = root
        path_to_leaf = [root.state]
        visited_positions = {node.state: 1}
        
        # Selection with pruning
        while node.is_fully_expanded(game_map, cached_moves) and node.children:
            node = node.best_child(exploration_factor, target)
            current_distance = manhattan_distance(node.state, target)
            
            # Cut if path is getting too long without making progress
            if (len(path_to_leaf) > max_reasonable_path_length and 
                current_distance >= node.parent.min_distance_to_target):
                break
                
            path_to_leaf.append(node.state)
            visited_positions[node.state] = visited_positions.get(node.state, 0) + 1
            
            if node.state == target:
                if len(path_to_leaf) < best_distance:
                    best_path = path_to_leaf.copy()
                    best_distance = len(path_to_leaf)
                break
        
        # Expansion with move selection
        if node.state != target:
            valid_moves = cached_moves.get(node.state) or get_valid_moves(game_map, node.state)
            cached_moves[node.state] = valid_moves
            unexplored = [move for move in valid_moves if move not in [child.state for child in node.children]]
            
            if unexplored:
                # Choose the most interesting unexplored move
                new_state = min(unexplored, 
                              key=lambda pos: manhattan_distance(pos, target) + 
                                            visited_positions.get(pos, 0))
                node = node.add_child(new_state, target)
                path_to_leaf.append(node.state)
                visited_positions[node.state] = visited_positions.get(node.state, 0) + 1
        
        # Simulation with path optimization
        current_state = node.state
        simulation_path = path_to_leaf.copy()
        simulation_steps = 0
        local_best_distance = manhattan_distance(current_state, target)
        
        while current_state != target and simulation_steps < max_reasonable_path_length:
            valid_moves = cached_moves.get(current_state) or get_valid_moves(game_map, current_state)
            cached_moves[current_state] = valid_moves
            
            if not valid_moves:
                break
            
            # move selection during simulation
            move_scores = []
            for move in valid_moves:
                distance_to_target = manhattan_distance(move, target)
                visit_count = visited_positions.get(move, 0)
                
                # Score based on multiple factors
                distance_score = 1.0 / (distance_to_target + 1)
                novelty_score = 1.0 / (visit_count + 1)
                progress_score = max(0, local_best_distance - distance_to_target)
                
                total_score = (distance_score + novelty_score + 0.5 * progress_score)
                move_scores.append(total_score)
            
            probabilities = np.array(move_scores)
            probabilities = probabilities / probabilities.sum()
            
            next_move = valid_moves[np.random.choice(len(valid_moves), p=probabilities)]
            current_state = next_move
            simulation_path.append(current_state)
            
            # Update tracking variables
            visited_positions[current_state] = visited_positions.get(current_state, 0) + 1
            local_best_distance = min(local_best_distance, manhattan_distance(current_state, target))
            simulation_steps += 1
        
        # Calculate reward with the improved function
        final_distance = manhattan_distance(current_state, target)
        reward = dynamic_reward(
            len(simulation_path),
            final_distance,
            initial_distance,
            best_distance,
            visited_positions.get(current_state, 0)
        )
        
        if current_state == target and len(simulation_path) < best_distance:
            best_path = simulation_path.copy()
            best_distance = len(simulation_path)
        
        # Backpropagation with distance tracking
        while node:
            node.visits += 1
            node.value += reward
            node.min_distance_to_target = min(node.min_distance_to_target, final_distance)
            node = node.parent
    
    return best_path if best_path and best_path[-1] == target else []
