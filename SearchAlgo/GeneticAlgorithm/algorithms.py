import numpy as np
from collections import deque
from queue import PriorityQueue
from utils import get_valid_moves
from typing import Tuple, List

def build_path(parent: dict, target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path

def bfs(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
    # Create a queue for BFS and mark the start node as visited
    queue = deque()
    visited = set()
    queue.append(start)
    visited.add(start)
    list_paths = []

    # Create a dictionary to keep track of the parent node for each node in the path
    parent = {start: None}

    #print(f"Starting BFS from {start} to {target}")
    #print(f"Initial queue: {list(queue)}")
    #print(f"Initial visited set: {visited}")

    while queue:
        # Dequeue a vertex from the queue
        current = queue.popleft()
        list_paths.append(build_path(parent, current))
        #print(f"\nDequeued: {current}")
        #print(f"Queue after dequeue: {list(queue)}")

        # Check if the target node has been reached
        if current == target:
            print("\nTarget found!")
            path = build_path(parent, target)
            #print(f"Path: {path}")
            #print(f"All paths: {list_paths}")
            list_paths.append(path)
            return list_paths

        # Visit all adjacent neighbors of the dequeued vertex
        for neighbor in get_valid_moves(game_map, current):
            #print(f"Valid neighbors of {current}: {neighbor}")
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current
                #print(f"Visited {neighbor}, added to queue. Updated queue: {list(queue)}")
                #print(f"Updated visited set: {visited}")

    print("Target node not found!")
    return None

# ---------------------------------------------

def bfs_uninformed(game_map: np.ndarray, start: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
    """
    Performs BFS on an uninformed map, exploring all reachable nodes from the start point.
    
    Parameters:
        game_map (np.ndarray): A 2D numpy array representing the game map.
        start (Tuple[int, int]): The starting coordinates.

    Returns:
        List[List[Tuple[int, int]]]: A list of paths explored during the BFS.
    """
    # Initialize the BFS queue and visited set
    queue = deque()
    visited = set()
    queue.append(start)
    visited.add(start)
    list_paths = []

    # Create a dictionary to keep track of the parent node for each node in the path
    parent = {start: None}

    while queue:
        # Dequeue a vertex from the queue
        current = queue.popleft()
        list_paths.append(build_path(parent, current))

        # Visit all adjacent neighbors of the dequeued vertex
        for neighbor in get_valid_moves(game_map, current):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current

    return list_paths

# ---------------------------------------------

def a_star(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable) -> List[Tuple[int, int]]:
    # initialize open and close list
    open_list = PriorityQueue()
    close_list = []
    # additional dict which maintains the nodes in the open list for an easier access and check
    support_list = {}

    starting_state_g = 0
    starting_state_h = h(start, target)
    starting_state_f = starting_state_g + starting_state_h

    open_list.put((starting_state_f, (start, starting_state_g)))
    support_list[start] = starting_state_g
    parent = {start: None}

    while not open_list.empty():
        # get the node with lowest f
        _, (current, current_cost) = open_list.get()
        # add the node to the close list
        close_list.append(current)

        if current == target:
            print("Target found!")
            path = build_path(parent, target)
            return path

        for neighbor in get_valid_moves(game_map, current):
            # check if neighbor in close list, if so continue
            if neighbor in close_list:
                continue
            # compute neighbor g, h and f values
            neighbor_g = 1 + current_cost
            neighbor_h = h(neighbor, target)
            neighbor_f = neighbor_g + neighbor_h
            parent[neighbor] = current
            neighbor_entry = (neighbor_f, (neighbor, neighbor_g))
            # if neighbor in open_list
            if neighbor in support_list.keys():
                # if neighbor_g is greater or equal to the one in the open list, continue
                if neighbor_g >= support_list[neighbor]:
                    continue
            
            # add neighbor to open list and update support_list
            open_list.put(neighbor_entry)
            support_list[neighbor] = neighbor_g

    print("Target node not found!")
    return None

# ----------------------------------------