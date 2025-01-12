# PSEDOCODE

'''
function MONTE-CARLO-TREE-SEARCH(state) returns an action
    tree <- NODE(state)
    while IS-TIME-REMAINING() do
        leaf <- SELECT(tree)
        child <- EXPAND(leaf)
        result <- SIMULATE(child)
        BACK-PROPAGATE(result, child)
    return the move in ACTIONS(state) whose node has highest number of playouts

'''

import numpy as np
import random

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self, possible_actions):
        explored_actions = {child.action for child in self.children}
        return len(explored_actions) == len(possible_actions)


    def best_child(self, exploration_weight=1.0):
        weights = [
            (child.value / child.visits if child.visits > 0 else 0) +
            exploration_weight * np.sqrt(np.log(self.visits) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[np.argmax(weights)]

    def expand(self, possible_actions, environment):
        unexplored_actions = [a for a in possible_actions if a not in {c.action for c in self.children}]
        if not unexplored_actions:
            return
        action = random.choice(unexplored_actions)
        try:
            next_state, reward, done, info = environment.step(action)
            child_node = Node(state=next_state, parent=self, action=action)
            self.children.append(child_node)
        except Exception as e:
            print(f"Error during expansion: {e}")


    
def rollout(environment, start_state, max_steps=50):
    state = environment.reset()  # Use the provided start state
    reward = 0

    for _ in range(max_steps):
        action = environment.action_space.sample()
        state, step_reward, done, _ = environment.step(action)
        reward += step_reward  # Accumulate reward during the rollout
        
        if done:
            break

    return reward

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent

def mcts(environment, start, target, max_iterations=1000, exploration_weight=1.0):
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def extract_player_position(state):
        # Assuming state is a dictionary with 'chars' key representing the game map
        game_map = np.array(state['chars']) if isinstance(state, dict) else state
        player_pos = np.where(game_map == ord('@'))
        return (player_pos[0][0], player_pos[1][0]) if player_pos[0].size > 0 else start

    environment.reset()
    root = Node(state=start)

    possible_actions = list(range(environment.action_space.n))

    for _ in range(max_iterations):
        node = root

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child(exploration_weight)

        # Expansion
        node.expand(possible_actions, environment)

        # Rollout with goal-directed reward
        try:
            current_pos = extract_player_position(node.state)
            reward = -manhattan_distance(current_pos, target)
        except Exception as e:
            print(f"Error extracting position: {e}")
            reward = 0

        # Backpropagation
        backpropagate(node, reward)

    # Path reconstruction
    path = [start]
    current_node = root

    while current_node.children and path[-1] != target:
        best_child = current_node.best_child(0)  # Exploitation
        
        # Determine next position based on action
        if best_child.action == 0:  # Move Up
            next_pos = (path[-1][0]-1, path[-1][1])
        elif best_child.action == 1:  # Move Down
            next_pos = (path[-1][0]+1, path[-1][1])
        elif best_child.action == 2:  # Move Left
            next_pos = (path[-1][0], path[-1][1]-1)
        elif best_child.action == 3:  # Move Right
            next_pos = (path[-1][0], path[-1][1]+1)
        else:
            break

        path.append(next_pos)
        current_node = best_child

        # Prevent infinite loops
        if len(path) > 50:  # Reasonable path length limit
            break

    return path
