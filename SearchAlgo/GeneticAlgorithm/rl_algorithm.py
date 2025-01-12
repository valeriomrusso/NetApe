import numpy as np
import time
from utils import *

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon_start=1.0):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action_size = action_size
        
    def get_action(self, state):
        state = tuple(map(tuple, state))  # Convert state to hashable format
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
            
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        state = tuple(map(tuple, state))
        next_state = tuple(map(tuple, next_state))
        
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)
            
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def calculate_reward(current_pos, target_pos, done, hp):
    """Calculate reward based on distance to target and game state"""
    if done and hp <= 0:  # Agent died
        return -100
    if done:  # Reached target
        return 100
        
    # Calculate distance-based reward
    prev_distance = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
    return -0.1 - (prev_distance * 0.1)  # Small negative reward for each step plus distance penalty

def train_rl_agent(env, episodes=1000, max_steps=100, print_interval=10):
    agent = QLearningAgent(state_size=2, action_size=4)  # 4 actions: up, down, left, right
    start_time = time.time()
    
    metrics = {
        'episodes_to_solve': 0,
        'success_rate': 0,
        'average_steps': 0,
        'best_episode_steps': float('inf')
    }
    
    successful_episodes = 0
    total_steps = 0
    
    print("Starting RL training...")
    print(f"Total episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    
    for episode in range(episodes):
        state = env.reset()
        current_pos = get_player_location(state['chars'])
        target_pos = get_target_location(state['chars'])
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            action = agent.get_action(state['chars'])
            next_state, _, done, _ = env.step(action)
            next_pos = get_player_location(next_state['chars'])
            hp = next_state['blstats'][6]
            
            reward = calculate_reward(next_pos, target_pos, done, hp)
            episode_reward += reward
            
            agent.update(state['chars'], action, reward, next_state['chars'])
            state = next_state
            steps += 1
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        if done and hp > 0:  # Successfully reached target
            successful_episodes += 1
            total_steps += steps
            metrics['best_episode_steps'] = min(metrics['best_episode_steps'], steps)
            if metrics['episodes_to_solve'] == 0:
                metrics['episodes_to_solve'] = episode + 1
        
        # Print progress
        if (episode + 1) % print_interval == 0:
            current_success_rate = successful_episodes / (episode + 1)
            elapsed_time = time.time() - start_time
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"Current epsilon: {agent.epsilon:.3f}")
            print(f"Success rate: {current_success_rate:.2%}")
            print(f"Best episode steps: {metrics['best_episode_steps']}")
            print(f"Episode reward: {episode_reward:.2f}")
            print(f"Current episode steps: {steps}")
            
    end_time = time.time()
    metrics['execution_time'] = end_time - start_time
    metrics['success_rate'] = successful_episodes / episodes
    metrics['average_steps'] = total_steps / successful_episodes if successful_episodes > 0 else float('inf')
    
    print("\nTraining completed!")
    print(f"Total time: {metrics['execution_time']:.2f} seconds")
    print(f"Final success rate: {metrics['success_rate']:.2%}")
    print(f"Average steps for successful episodes: {metrics['average_steps']:.2f}")
    print(f"Best episode steps: {metrics['best_episode_steps']}")
    
    return metrics, agent