import numpy as np
import random
from utils import get_valid_moves
from typing import Tuple, List, Dict, Set

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state # current state represented by the node
        self.parent = parent # reference to the parent node
        self.children = [] # list of child nodes
        self.visits = 0 # number of times this node has been visited
        self.value = 0 # cumulative reward associated with this node
        # orthogonal moves checking
        if parent is None:
            self.path_to_leaf = [state]
        else:
            
            px, py = parent.state
            cx, cy = state
            if abs(px - cx) + abs(py - cy) != 1:  # Verifica che la mossa sia di un solo passo in una direzione
                raise ValueError(f"Invalid move from {parent.state} to {state}")
            self.path_to_leaf = parent.path_to_leaf + [state]
            
        self.min_distance_to_target = float('inf')
        self.visited_states = set() if parent is None else parent.visited_states.copy()
        self.visited_states.add(state)
        
    def is_fully_expanded(self, game_map: np.ndarray, cached_moves: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> bool:
        if self.state not in cached_moves:
            cached_moves[self.state] = get_valid_moves(game_map, self.state)
        return len(self.children) == len([m for m in cached_moves[self.state] if m not in self.visited_states])

    def best_child(self, exploration_weight: float = 1.0, target: Tuple[int, int] = None):
        if not self.children:
            return None
            
        scores = []
        for child in self.children:
            exploitation = child.value / (child.visits + 1e-6)
            exploration = exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
            
            if target:
                current_distance = manhattan_distance(child.state, target)
                distance_score = 1.0 / (current_distance + 1)
                path_length_penalty = 0.2 * len(child.path_to_leaf)
                revisit_penalty = 0.3 * len(child.visited_states.intersection(self.visited_states))
                
                total_score = exploitation + exploration + distance_score - path_length_penalty - revisit_penalty
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
                  best_distance: float, visited_count: int, cycle_detected: bool) -> float:
    if current_distance == 0:  # Target reached
        path_efficiency = max(0, 1 - (path_length / (initial_distance * 2)))
        return 2.0 + path_efficiency  # Increased reward for reaching target
        
    # Heavily penalize paths with loops
    if cycle_detected:
        return 0.05
        
    progress = (initial_distance - current_distance) / initial_distance
    length_penalty = 0.3 * (path_length / initial_distance)  # Increased length penalty
    revisit_penalty = 0.2 * visited_count  # Increased revisit penalty
    
    return max(0.1, progress - length_penalty - revisit_penalty)

def mcts(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], 
         iterations: int = 1000, exploration_factor: float = 0.3) -> List[Tuple[int, int]]:
    
    root = MCTSNode(start)
    best_path = []
    best_distance = float('inf')
    cached_moves = {}
    
    initial_distance = manhattan_distance(start, target)
    max_reasonable_path_length = initial_distance * 3  # Increased max path length
    
    for iteration in range(iterations):
        node = root
        current_path = set([start])
        
        # Selection with cycle detection
        while node.is_fully_expanded(game_map, cached_moves) and node.children:
            node = node.best_child(exploration_factor, target)
            if node.state in current_path:  # Cycle detected
                break
            current_path.add(node.state)
            
            if node.state == target:
                if len(node.path_to_leaf) < best_distance:
                    best_path = node.path_to_leaf.copy()
                    best_distance = len(node.path_to_leaf)
                break
        
        # Expansion
        if node.state != target:
            valid_moves = cached_moves.get(node.state) or get_valid_moves(game_map, node.state)
            cached_moves[node.state] = valid_moves
            unexplored = [move for move in valid_moves if move not in [child.state for child in node.children] 
                         and move not in node.visited_states]
            
            if unexplored:
                # Prioritize moves that haven't been visited and are closer to target
                new_state = min(unexplored, 
                              key=lambda pos: manhattan_distance(pos, target) + 
                                            len(current_path & {pos}) * 10)
                node = node.add_child(new_state, target)
                current_path.add(new_state)
        
        # Simulation with improved path finding
        current_state = node.state
        simulation_path = set(current_path)
        simulation_steps = 0
        cycle_detected = False
        
        while current_state != target and simulation_steps < max_reasonable_path_length:
            valid_moves = cached_moves.get(current_state) or get_valid_moves(game_map, current_state)
            cached_moves[current_state] = valid_moves
            
            if not valid_moves:
                break
                
            # Improved move selection during simulation
            move_scores = []
            for move in valid_moves:
                distance_to_target = manhattan_distance(move, target)
                visit_penalty = 2.0 if move in simulation_path else 0.0
                progress_score = (manhattan_distance(current_state, target) - distance_to_target)
                
                score = progress_score - visit_penalty + random.random() * 0.1
                move_scores.append(score)
            
            next_move = valid_moves[np.argmax(move_scores)]
            
            if next_move in simulation_path:
                cycle_detected = True
                break
                
            current_state = next_move
            simulation_path.add(current_state)
            simulation_steps += 1
        
        # Calculate reward with improved cycle detection
        final_distance = manhattan_distance(current_state, target)
        reward = dynamic_reward(
            len(simulation_path),
            final_distance,
            initial_distance,
            best_distance,
            len(simulation_path),
            cycle_detected
        )
        
        # Backpropagation
        while node:
            node.visits += 1
            node.value += reward
            node.min_distance_to_target = min(node.min_distance_to_target, final_distance)
            node = node.parent
            
    return best_path if best_path and best_path[-1] == target else []