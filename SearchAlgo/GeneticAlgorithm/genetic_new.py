import numpy as np
import random
from utils import get_valid_moves
from typing import Tuple, List

# Funzioni ausiliarie

def fitness(path: List[Tuple[int, int]], target: Tuple[int, int]) -> float:
    """
    Calcola il fitness di un percorso basandosi sulla distanza dal target.
    Percorsi più vicini al target ottengono un punteggio più alto.
    """
    if not path:  # Penalizza percorsi vuoti
        return float('-inf')
    last_node = path[-1]  # Ultimo nodo del percorso
    # Distanza Manhattan dal target
    dist = abs(last_node[0] - target[0]) + abs(last_node[1] - target[1])
    return -dist  # Distanza negativa per favorire percorsi più vicini

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

def crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Combina due percorsi genitori per creare un nuovo percorso figlio.
    """
    # Scegli un punto di divisione casuale
    split = random.randint(1, min(len(parent1), len(parent2)) - 1)
    # Combina i primi segmenti di parent1 con il resto di parent2
    child = parent1[:split] + parent2[split:]
    return child

def mutate(path: List[Tuple[int, int]], game_map: np.ndarray, mutation_rate: float) -> List[Tuple[int, int]]:
    """
    Introduce una mutazione casuale in un percorso.
    Cambia una posizione del percorso con una mossa valida casuale.
    """
    if not path:
        return path  # Percorso vuoto, nessuna mutazione
    if random.random() < mutation_rate:  # Verifica se applicare una mutazione
        idx = random.randint(0, len(path) - 1)  # Scegli una posizione casuale
        neighbors = get_valid_moves(game_map, path[idx])  # Ottieni mosse valide per quella posizione
        if neighbors:
            path[idx] = random.choice(neighbors)  # Modifica la posizione con una mossa valida
    return path

# Funzione principale

def genetic_alg_func(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], 
                      population_size: int, generations: int, mutation_rate: float, max_steps: int) -> List[List[Tuple[int, int]]]:
    
    # **Popolazione iniziale**: Genera un insieme di percorsi casuali
    population = [generate_random_path(game_map, start, target, max_steps) for _ in range(population_size)]
    list_paths = []  # Lista per salvare i percorsi di ogni generazione

    for generation in range(generations):  # Itera attraverso le generazioni
        # **Valutazione**: Ordina la popolazione in base al fitness (decrescente)
        population.sort(key=lambda path: fitness(path, target), reverse=True)

        # Salva il miglior percorso della generazione
        best_path = population[0]
        list_paths.append(best_path)
        
        # Se il miglior percorso raggiunge il target, termina
        if best_path and best_path[-1] == target:
            print(f"Target raggiunto nella generazione {generation}")
            return list_paths

        # **Selezione**: Mantieni i migliori percorsi (metà della popolazione)
        population = population[:population_size // 2]

        # **Riproduzione**: Genera una nuova popolazione con crossover e mutazione
        new_population = []
        while len(new_population) < population_size:
            # Seleziona due genitori casualmente
            parent1, parent2 = random.sample(population, 2)
            # Crea un figlio tramite crossover
            child = crossover(parent1, parent2)
            # Applica mutazione al figlio
            child = mutate(child, game_map, mutation_rate)
            new_population.append(child)

        # Aggiorna la popolazione
        population = new_population

    print("Target non raggiunto dopo tutte le generazioni.")
    return list_paths
