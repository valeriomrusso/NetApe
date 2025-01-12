import numpy as np
import random
from utils import get_valid_moves
from typing import Tuple, List, Optional

def build_path(parent: dict, target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    while target is not None:
        path.append(target)
        if target in parent:  # Verifica se il target è presente nel dizionario
            target = parent[target]
        else:
            break  # Se il target non è nel dizionario, interrompe il ciclo
    path.reverse()
    return path

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state # posizione del giocatore sulla mappa
        self.parent = parent
        self.children = [] # lista dei nodi figli generati da questo nodo
        self.visits = 0 # numero di volte che il nodo è stato visitato durante la ricerca
        self.value = 0 # somma cumulativa delle ricompense ottenute durante la propagazione all'indietro

    # true se il nodo è completamente espanso
    def is_fully_expanded(self, game_map: np.ndarray) -> bool:
        # controlla se tutte le mosse valide sono state esplorate aggiungendo nodi figli
        return len(self.children) == len(get_valid_moves(game_map, self.state))

    def best_child(self, exploration_weight: float = 1.0):
        # Usa UCT (upper confidence bound for trees) per selezionare il miglior figlio
        return max(
            self.children,
            # exploration weight: controlla il bilanciamento tra sfruttamento (alta ricompensa) e esplorazione (mosse poco visitate)
            key=lambda child: child.value / (child.visits + 1e-6) + 
            (exploration_weight * np.sqrt(
                np.log(self.parent.visits + 1) / (child.visits + 1e-6)
            ) if self.parent is not None else 0)
        )

    # crea un nuovo figlio con lo stato fornito e lo aggiunge alla lista dei figli del nodo corrente
    def add_child(self, child_state):
        child_node = MCTSNode(child_state, parent=self)
        self.children.append(child_node)
        return child_node

def mcts(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], iterations: int = 1000) -> List[Tuple[int, int]]:
    """
    Monte Carlo Tree Search di base per trovare un percorso verso il target.
    """
    root = MCTSNode(start)
    best_path = []
    best_distance = float('inf')

    for _ in range(iterations):
        # Selection & Expansion
        node = root
        path_to_leaf = [node.state]  # Tiene traccia del percorso durante la selezione
        
        # Continua la selezione finché non troviamo un nodo non completamente espanso
        while node.is_fully_expanded(game_map) and node.children:
            node = node.best_child()
            path_to_leaf.append(node.state)
            
            # Se abbiamo raggiunto il target, interrompi
            if node.state == target:
                if len(path_to_leaf) < best_distance:
                    best_path = path_to_leaf.copy()
                    best_distance = len(path_to_leaf)
                break
        
        # Se il nodo non è completamente espanso, espandilo
        if node.state != target:
            valid_moves = get_valid_moves(game_map, node.state)
            unexplored = [move for move in valid_moves 
                         if move not in [child.state for child in node.children]]
            
            if unexplored:  # Se ci sono mosse inesplorate
                new_state = random.choice(unexplored)
                node = node.add_child(new_state)
                path_to_leaf.append(node.state)
        
        # Simulation
        current_state = node.state
        simulation_path = path_to_leaf.copy()
        simulation_steps = 0
        max_simulation_steps = 100  # Limite per evitare simulazioni infinite
        
        while current_state != target and simulation_steps < max_simulation_steps:
            valid_moves = get_valid_moves(game_map, current_state)
            if not valid_moves:
                break
                
            # Scegli la mossa che si avvicina di più al target
            next_state = min(valid_moves, 
                           key=lambda pos: abs(pos[0] - target[0]) + abs(pos[1] - target[1]))
            current_state = next_state
            simulation_path.append(current_state)
            simulation_steps += 1
        
        # Calcola la ricompensa
        if current_state == target:
            reward = 1.0
            if len(simulation_path) < best_distance:
                best_path = simulation_path.copy()
                best_distance = len(simulation_path)
        else:
            # Ricompensa parziale basata sulla distanza dal target
            final_distance = abs(current_state[0] - target[0]) + abs(current_state[1] - target[1])
            initial_distance = abs(start[0] - target[0]) + abs(start[1] - target[1])
            reward = 0.5 * (initial_distance - final_distance) / initial_distance
        
        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    # Se abbiamo trovato un percorso valido, usalo
    if best_path and best_path[-1] == target:
        return best_path
        
    # Altrimenti, costruisci il percorso dal miglior figlio
    best_child = root.best_child(exploration_weight=0)
    return build_path({child.state: child.parent.state for child in root.children}, best_child.state)



def simulate_exploration_with_probabilities(state: Tuple[int, int], target: Tuple[int, int], game_map: np.ndarray, exploration_factor: float = 0.2,local_maxima_penalty: float = 0.3,history_weight: float = 0.15) -> int:
    """
    Esegue una simulazione bilanciando:
    1. Avvicinamento al target
    2. Esplorazione di percorsi alternativi
    3. Evitamento di massimi locali attraverso penalità per revisitazione
    """
    current = state
    visited_positions = {current: 1}
    max_steps = game_map.shape[0] * game_map.shape[1] * 2
    steps = 0
    
    while current != target and steps < max_steps:
        valid_moves = get_valid_moves(game_map, current)
        valid_moves = [move for move in valid_moves if move != current]
        
        if not valid_moves:
            return 0
            
        # Calcola componenti multiple per il punteggio di ogni mossa
        scores = []
        for move in valid_moves:
            # 1. Distanza dal target (normalizzata e invertita)
            distance = abs(move[0] - target[0]) + abs(move[1] - target[1])
            distance_score = 1.0 / (distance + 1)
            
            # 2. Penalità per posizioni già visitate
            visit_penalty = local_maxima_penalty * visited_positions.get(move, 0)
            
            # 3. Componente di "novità" basata sulla storia delle visite
            novelty_score = 1.0 / (sum(abs(move[0] - pos[0]) + abs(move[1] - pos[1]) <= 2 
                                     for pos in visited_positions) + 1)
            
            # Combina i punteggi
            total_score = (
                distance_score * (1 - history_weight - exploration_factor) +
                novelty_score * history_weight -
                visit_penalty
            )
            scores.append(max(0.01, total_score))
        
        # Normalizza i punteggi in probabilità
        scores = np.array(scores)
        base_probabilities = scores / np.sum(scores)
        
        # Aggiungi componente di esplorazione casuale
        move_probabilities = (1 - exploration_factor) * base_probabilities + \
                           exploration_factor * np.ones_like(base_probabilities) / len(base_probabilities)
        
        # Normalizza le probabilità finali
        move_probabilities /= np.sum(move_probabilities)
        
        # Seleziona la prossima mossa 
        # Mantieni valid_moves come lista di tuple e seleziona direttamente da essa
        chosen_index = np.random.choice(len(valid_moves), p=move_probabilities)
        chosen_move = valid_moves[chosen_index]
        
        # Aggiorna la storia delle visite
        visited_positions[chosen_move] = visited_positions.get(chosen_move, 0) + 1
        
        current = chosen_move
        steps += 1
    
    return 1 if current == target else 0


def backpropagate(node: MCTSNode, reward: int):
    """
    Propaga il risultato della simulazione indietro lungo l'albero.
    """
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent


