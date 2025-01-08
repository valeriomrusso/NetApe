import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import get_valid_moves
from typing import Tuple, List

"""
####################################################
## Pseudocode for gbfs (greedy best first search) ##
####################################################

procedure gbfs:
    Mark start as visited
    add start to queue
    while queue is not empty:
        current_node <- the node with the closest distance to the target
        remove current_node from queue
        for each neighbor n of current_node:
            if n is not visited:
                if n is target:
                    return current node
                else:
                    mark n as visited
                    add n to queue
            
return failure

"""


def __build_path(parent: dict, target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path


def greedy_best_first_search(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable):
    queue = PriorityQueue()
    visited = set()

    queue.put((h(start, target), start))
    visited.add(start)

    parent = {start: None}
    
    while not queue.empty():
        # Get the node with the smallest heuristic value
        _, current_node = queue.get()

        if current_node == target:
            print("target found")
            path = __build_path(parent, target)
            return path

        for neighbor in get_valid_moves(game_map, current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put((h(neighbor, target), neighbor))
                parent[neighbor] = current_node

    print("Target not reachable")
    return None