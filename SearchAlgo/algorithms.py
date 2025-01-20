import numpy as np

from collections import deque
from queue import PriorityQueue

from utils import euclidean_distance, is_wall, monster_penalty, get_valid_moves
from typing import Tuple, List
import heapq

def build_path(parent: dict, target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path

def bfs(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    # Create a queue for BFS and mark the start node as visited
    queue = deque()
    visited = set()
    queue.append(start)
    visited.add(start)

    # Create a dictionary to keep track of the parent node for each node in the path
    parent = {start: None}

    while queue:
        # Dequeue a vertex from the queue
        current = queue.popleft()

        # Check if the target node has been reached
        if current == target:
            print("Target found!")
            path = build_path(parent, target)
            return path

        # Visit all adjacent neighbors of the dequeued vertex
        for neighbor in get_valid_moves(game_map, current):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current

    print("Target node not found!")
    return None

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

def greedy(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable) -> List[Tuple[int, int]]:
    open_list = PriorityQueue()
    open_list.put((h(start, target), start))

    visited = set()
    parent = {start: None}

    while not open_list.empty():
        _, current = open_list.get()

        if current == target:
            print("Target found!")
            return build_path(parent, target)

        visited.add(current)

        for neighbor in get_valid_moves(game_map, current):
            if neighbor not in visited:
                open_list.put((h(neighbor, target), neighbor))
                visited.add(neighbor)
                parent[neighbor] = current

    print("Target node not found!")
    return None

# ----------------------------------------

def dfs(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    stack = []
    visited = set()
    stack.append(start)
    visited.add(start)

    parent = {start: None}

    while stack:
        current = stack.pop()

        if current == target:
            print("Target found!")
            return build_path(parent, target)

        for neighbor in get_valid_moves(game_map, current):
            if neighbor not in visited:
                stack.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current

    print("Target node not found!")
    return None

# ----------------------------------------

# Iterative Deepening Depth-First
def iddfs(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], max_depth: int) -> List[Tuple[int, int]]:

    for depth in range(max_depth + 1):
        print(f"Exploring depth: {depth}")
        path = []
        visited = set()
        if dfs_limited(game_map, start, target, depth, path, visited):
            return path
    print("Target node not found!")
    return None

# DFS fino a un limite di profondità specificato
def dfs_limited(game_map: np.ndarray, current: Tuple[int, int], target: Tuple[int, int], depth: int,
                path: List[Tuple[int, int]], visited: set) -> bool:

    if depth == 0 and current == target:
        path.append(current)
        return True
    if depth > 0:
        visited.add(current)
        for neighbor in get_valid_moves(game_map, current):
            if neighbor not in visited:
                if dfs_limited(game_map, neighbor, target, depth - 1, path, visited):
                    path.append(current)
                    return True
        visited.remove(current)
    return False

# ----------------------------------------

def beam_search(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable, beam_width: int) -> List[Tuple[int, int]]:
    current_nodes = [start]
    parent = {start: None}
    visited = set()

    while current_nodes:
        neighbors = []

        for node in current_nodes:
            if node == target:
                print("Target found!")
                return build_path(parent, target)

            visited.add(node)

            for neighbor in get_valid_moves(game_map, node):
                if neighbor not in visited:
                    neighbors.append((h(neighbor, target), neighbor))
                    parent[neighbor] = node

        if not neighbors:
            break

        neighbors.sort()
        current_nodes = [node for _, node in neighbors[:beam_width]]

    print("Target node not found!")
    return None

# ----------------------------------------

def theta_star(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable) -> List[Tuple[int, int]]:
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: start}
    g_score = {start: 0}
    visited = set()

    while open_list:
        _, current = heapq.heappop(open_list)
        visited.add(current)

        if current == target:
            return build_path(came_from, target)

        for neighbor in get_valid_moves(game_map, current):
            if neighbor in visited:
                continue

            parent = came_from.get(current, None)
            if parent is None:
                parent = current

            if line_of_sight(game_map, parent, neighbor):
                tentative_g_score = g_score[parent] +  euclidean_distance(parent, neighbor)
            else:
                tentative_g_score = g_score[current] + euclidean_distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + h(neighbor, target)
                heapq.heappush(open_list, (f_score, neighbor))

    print("Target node not found!")
    return None

def line_of_sight(game_map: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> bool:

    #Determina se c'è una linea di vista diretta tra due punti senza ostacoli.
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
    err = dx - dy

    while (x0, y0) != (x1, y1):
        if is_wall(game_map[x0, y0]):
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return True

# ----------------------------------------

def a_star_with_monsters(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], monsters: List[Tuple[int, int]], h: callable) -> List[Tuple[int, int]]:
    open_list = PriorityQueue()
    closed_list = set()
    support_list = {}

    starting_state_g = 0
    starting_state_h = h(start, target)
    starting_state_f = starting_state_g + starting_state_h

    open_list.put((starting_state_f, (start, starting_state_g)))
    support_list[start] = starting_state_g
    parent = {start: None}

    while not open_list.empty():
        _, (current, current_cost) = open_list.get()
        closed_list.add(current)

        if current == target:
            print("Target found!")
            path = build_path(parent, target)
            return path

        for neighbor in get_valid_moves(game_map, current):
            if neighbor in closed_list:
                continue

            # Calcola il costo g e h
            neighbor_g = 1 + current_cost + monster_penalty(game_map, neighbor, monsters)
            neighbor_h = h(neighbor, target)
            neighbor_f = neighbor_g + neighbor_h
            parent[neighbor] = current

            if neighbor in support_list:
                if neighbor_g >= support_list[neighbor]:
                    continue

            open_list.put((neighbor_f, (neighbor, neighbor_g)))
            support_list[neighbor] = neighbor_g

    print("Target node not found!")
    return None
