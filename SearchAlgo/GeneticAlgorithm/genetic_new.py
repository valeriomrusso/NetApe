import numpy as np
import random
from utils import get_valid_moves
from typing import Tuple, List

# Funzioni ausiliarie

def fitness(path: List[Tuple[int, int]], target: Tuple[int, int]) -> float:
    if not path:
        return float('-inf')
    
    last_node = path[-1]
    dist = abs(last_node[0] - target[0]) + abs(last_node[1] - target[1])
    
    # Penalizza ripetizioni
    position_counts = {}
    for pos in path:
        position_counts[pos] = position_counts.get(pos, 0) + 1
    
    repetition_penalty = sum(count - 1 for count in position_counts.values())
    
    # Calcola progressione verso il target
    unique_positions = len(set(path))
    progress_score = unique_positions / len(path)
    
    return -(dist + repetition_penalty * 2) + (progress_score * 10)

def generate_random_path(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], max_steps: int) -> List[Tuple[int, int]]:
    """
    Genera un percorso casuale partendo dal nodo iniziale.
    Il percorso si sviluppa scegliendo mosse valide fino a raggiungere il target o il limite massimo di passi.
    """
    path = [start]  # Inizia dal nodo di partenza
    current = start
    for _ in range(max_steps):  # Limita la lunghezza dei percorsi
        neighbors = get_valid_moves(game_map, current)  # Ottieni mosse valide
        if not neighbors:  # Se non ci sono mosse valide, termina il percorso
            break
        current = random.choice(neighbors)  # Scegli una mossa casuale
        path.append(current)
        if current == target:  # Se raggiunge il target, interrompi
            break
    return path

def is_valid_path(path: List[Tuple[int, int]]) -> bool:
    """
    Verifica che il percorso sia valido controllando che ogni passo sia adiacente al precedente.
    Un passo è valido se si muove di una sola unità in orizzontale o verticale.
    """
    if not path or len(path) < 2:
        return True
        
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        
        # Verifica che il movimento sia di una sola unità in orizzontale o verticale
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        
        # Un passo è valido se:
        # - si muove di 1 in orizzontale e 0 in verticale
        # - si muove di 0 in orizzontale e 1 in verticale
        if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
            return False
            
    return True

def crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Combina due percorsi genitori per creare un nuovo percorso figlio,
    assicurandosi che il punto di split produca un percorso valido.
    """
    if len(parent1) < 3 or len(parent2) < 3:
        return random.choice([parent1, parent2])
    
    max_split = min(len(parent1), len(parent2)) - 1
    for _ in range(max_split):  # Tenta più volte di trovare uno split valido
        split = random.randint(1, max_split)  # Scegli un punto di split casuale
        
        # Verifica se il collegamento è valido (movimento di una unità in verticale o orizzontale)
        if (
            (abs(parent1[split - 1][0] - parent2[split][0]) == 1 and parent1[split - 1][1] == parent2[split][1]) or
            (parent1[split - 1][0] == parent2[split][0] and abs(parent1[split - 1][1] - parent2[split][1]) == 1)
        ) and parent1[split - 1] != parent2[split]:
            return parent1[:split] + parent2[split:]
    
    # Se non trova uno split valido, ritorna uno dei genitori
    return random.choice([parent1, parent2])

def mutate(path: List[Tuple[int, int]], game_map: np.ndarray, mutation_rate: float) -> List[Tuple[int, int]]:
    """
    Introduce una mutazione casuale in un percorso,
    mantenendo la validità dei movimenti.
    """
    if not path:
        return path
    
    if random.random() < mutation_rate:
        idx = random.randint(0, len(path) - 1)
        neighbors = get_valid_moves(game_map, path[idx])
        
        if neighbors:
            new_step = random.choice(neighbors)
            
            # Verifica che la nuova mossa sia valida rispetto al passo precedente e successivo
            is_valid = True
            if idx > 0:
                is_valid = is_valid and (
                    (abs(new_step[0] - path[idx - 1][0]) == 1 and new_step[1] == path[idx - 1][1]) or
                    (new_step[0] == path[idx - 1][0] and abs(new_step[1] - path[idx - 1][1]) == 1)
                )
            
            if idx < len(path) - 1:
                is_valid = is_valid and (
                    (abs(new_step[0] - path[idx + 1][0]) == 1 and new_step[1] == path[idx + 1][1]) or
                    (new_step[0] == path[idx + 1][0] and abs(new_step[1] - path[idx + 1][1]) == 1)
                )
            
            if is_valid:
                path[idx] = new_step
    
    return path


# Funzione principale
def genetic_alg_func(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], 
                    population_size: int, generations: int, mutation_rate: float, max_steps: int) -> List[List[Tuple[int, int]]]:
    
    # Popolazione iniziale
    population = [generate_random_path(game_map, start, target, max_steps) for _ in range(population_size)]
    list_paths = []  # Lista per salvare i percorsi di ogni generazione
    
    # Tiene traccia del miglior fitness per rilevare stallo
    best_fitness_count = 0
    last_best_fitness = float('-inf')
    
    for generation in range(generations):
        # Valutazione
        population.sort(key=lambda path: fitness(path, target), reverse=True)
        
        # Salva il miglior percorso della generazione
        best_path = population[0]
        list_paths.append(best_path)
        
        # Controlla se abbiamo raggiunto il target
        if best_path and best_path[-1] == target:
            print(f"Target raggiunto nella generazione {generation}")
            return list_paths
            
        # Controlla stallo
        current_best_fitness = fitness(best_path, target)
        if abs(current_best_fitness - last_best_fitness) < 0.001:
            best_fitness_count += 1
        else:
            best_fitness_count = 0
        last_best_fitness = current_best_fitness
        
        # Se siamo in stallo, rigenera parte della popolazione
        if best_fitness_count > 10:
            elite_size = population_size // 10  # Mantieni solo il 10% migliore
            elite = population[:elite_size]
            new_random = [generate_random_path(game_map, start, target, max_steps) 
                         for _ in range(population_size - elite_size)]
            population = elite + new_random
            best_fitness_count = 0
            continue
            
        # Selezione con diversità
        elite_size = population_size // 4
        random_size = population_size // 4
        
        # Mantieni l'élite
        elite = population[:elite_size]
        # Mantieni alcuni individui casuali per la diversità
        random_selection = random.sample(population[elite_size:], random_size)
        # Genera nuova popolazione
        new_population = elite + random_selection
        
        # Riproduzione per completare la popolazione
        while len(new_population) < population_size:
            # Seleziona due genitori diversi
            parent1, parent2 = random.sample(population[:population_size//2], 2)
            # Crea un figlio tramite crossover
            child = crossover(parent1, parent2)
            # Applica mutazione
            child = mutate(child, game_map, mutation_rate)
            # Verifica che il figlio sia diverso dai genitori
            if child != parent1 and child != parent2:
                new_population.append(child)
        
        # Aggiorna la popolazione
        population = new_population[:population_size]  # Assicura dimensione corretta
    
    print("Target non raggiunto dopo tutte le generazioni.")
    return list_paths
