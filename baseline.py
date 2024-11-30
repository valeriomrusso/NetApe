# Install the package below first  
############################################
# pip install nle gymnasium ray[rllib]
############################################


import gymnasium as gym
import nle
import torch
import torch.nn as nn
from ray.rllib.algorithms.impala import ImpalaConfig

# Define the neural network architecture
class NetHackPolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(NetHackPolicy, self).__init__()
        
        # Observation space shapes
        self.screen_shape = observation_space["blstats"].shape[0]
        self.dungeon_shape = observation_space["chars"].shape
        
        # CNN for processing the dungeon map
        self.cnn_dungeon = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.dungeon_features = 32 * self.dungeon_shape[0] * self.dungeon_shape[1]

        # CNN for processing the screen
        self.cnn_screen = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.screen_features = 32 * self.screen_shape

        # MLP for structured data (e.g., stats, inventory)
        self.mlp_stats = nn.Sequential(
            nn.Linear(observation_space["blstats"].shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(input_size=self.dungeon_features + self.screen_features + 64, hidden_size=256)

        # Fully connected layer for policy output
        self.fc = nn.Linear(256, action_space.n)

    def forward(self, obs, hx, cx):
        # Process dungeon map
        dungeon_input = obs["chars"].unsqueeze(1).float()  # Add channel dimension
        dungeon_out = self.cnn_dungeon(dungeon_input)

        # Process screen
        screen_input = obs["chars"].unsqueeze(1).float()
        screen_out = self.cnn_screen(screen_input)

        # Process stats
        stats_input = obs["blstats"].float()
        stats_out = self.mlp_stats(stats_input)

        # Combine features
        combined = torch.cat([dungeon_out, screen_out, stats_out], dim=1)
        lstm_out, (hx, cx) = self.lstm(combined.unsqueeze(0), (hx, cx))

        # Policy output
        policy = self.fc(lstm_out.squeeze(0))
        return policy, (hx, cx)


# Define the environment wrapper for NLE
def create_env():
    env = gym.make("NetHackScore-v0")
    return env


# Train using Ray RLlib and IMPALA
def train_agent():
    config = ImpalaConfig().environment(
        env=create_env
    ).framework(
        framework="torch"
    ).training(
        model={"custom_model": NetHackPolicy}
    )

    # Instantiate the IMPALA trainer
    trainer = config.build()

    # Train the agent
    for i in range(10):
        results = trainer.train()
        print(f"Iteration {i}: Reward: {results['episode_reward_mean']}")

    # Save the trained policy
    trainer.save("nethack_agent")


if __name__ == "__main__":
    train_agent()
