
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
        self.path_to_leaf = [] if parent is None else parent.path_to_leaf + [state] # path from the root to this node

    def is_fully_expanded(self, game_map: np.ndarray, cached_moves: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> bool:
        # check if all possible moves from this state have been explored
        # uses cached valid moves for efficiency 
        
        if self.state not in cached_moves:
            # cache valid moves for the current state to avoid recomputation
            cached_moves[self.state] = get_valid_moves(game_map, self.state)
        return len(self.children) == len(cached_moves[self.state])

    def best_child(self, exploration_weight: float = 1.0):
        # select the best child node using upper confidence bound
        # balances exploitation and exploration
        return max(
            self.children,
            key=lambda child: (
                child.value / (child.visits + 1e-6) +
                (exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6)) if self.parent else 0)
            )
        )
    def add_child(self, child_state):
        # create and add a new child node for the given state

        child_node = MCTSNode(child_state, parent=self)
        self.children.append(child_node)
        return child_node


def dynamic_reward(path_length: int, best_distance: int, initial_distance: int, final_distance: int) -> float:
    '''
    calculate a reward for the simulation based on progress toward the target
    - full reward (1.0) for reaching the target
    - partial reward based on improvement in distance to the target
    - penalty for longer paths when no improvement is made
    '''

    if final_distance == 0:
        return 1.0  # Target raggiunto
    improvement = initial_distance - final_distance
    return 0.5 * (improvement / initial_distance) if improvement > 0 else -0.1 * (path_length - best_distance)


def mcts( game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], iterations: int = 1000, exploration_factor: float = 0.2) -> List[Tuple[int, int]]:
    # Monte Carlo Tree Search to find a path from start to target

    root = MCTSNode(start)
    best_path, best_distance = [], float('inf')
    cached_moves = {} # cache valid moved for efficiency 

    for _ in range(iterations):
        # Selection & Expansion
        node, path_to_leaf = root, [root.state]
        visited_positions = {node.state: 1}

        # traverse the tree until a non-fully-expanded node or target is found
        while node.is_fully_expanded(game_map, cached_moves) and node.children:
            node = node.best_child(exploration_weight=exploration_factor)
            path_to_leaf.append(node.state)
            visited_positions[node.state] = visited_positions.get(node.state, 0) + 1
            if node.state == target:
                # update best path if the target is reached
                if len(path_to_leaf) < best_distance:
                    best_path, best_distance = path_to_leaf.copy(), len(path_to_leaf)
                break

        if node.state != target:
            # expand the node by adding a new child 
            valid_moves = cached_moves.get(node.state) or get_valid_moves(game_map, node.state)
            cached_moves[node.state] = valid_moves
            unexplored = [move for move in valid_moves if move not in [child.state for child in node.children]]

            if unexplored:
                new_state = random.choice(unexplored)
                node = node.add_child(new_state)
                path_to_leaf.append(node.state)
                visited_positions[node.state] = visited_positions.get(node.state, 0) + 1

        # Simulation
        current_state, simulation_path = node.state, path_to_leaf.copy()
        max_simulation_steps, simulation_steps = 100, 0
        initial_distance = abs(start[0] - target[0]) + abs(start[1] - target[1])

        while current_state != target and simulation_steps < max_simulation_steps:
            
            # randomly simulate moves from the current state until the target is reached or steps are exhausted
            valid_moves = cached_moves.get(current_state) or get_valid_moves(game_map, current_state)
            cached_moves[current_state] = valid_moves
            if not valid_moves:
                break

            # compute probabilities for each move based on distance, penalties, and bonuses
            move_scores = []
            for move in valid_moves:
                distance = abs(move[0] - target[0]) + abs(move[1] - target[1])
                visit_penalty = 0.3 * visited_positions.get(move, 0)
                exploration_bonus = 1.0 / (sum(visited_positions.get(neighbor, 0) for neighbor in valid_moves) + 1)
                score = (1 / (distance + 1)) - visit_penalty + exploration_bonus
                move_scores.append(max(0.01, score))

            probabilities = np.array(move_scores)
            probabilities = probabilities / probabilities.sum()

            next_move = valid_moves[np.random.choice(len(valid_moves), p=probabilities)]
            visited_positions[next_move] = visited_positions.get(next_move, 0) + 1
            current_state = next_move
            simulation_path.append(current_state)
            simulation_steps += 1

        # calculate reward based on simulation results
        final_distance = abs(current_state[0] - target[0]) + abs(current_state[1] - target[1])
        reward = dynamic_reward(len(simulation_path), best_distance, initial_distance, final_distance)

        if current_state == target and len(simulation_path) < best_distance:
            best_path, best_distance = simulation_path.copy(), len(simulation_path)

        # Backpropagation
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    return best_path if best_path and best_path[-1] == target else []



