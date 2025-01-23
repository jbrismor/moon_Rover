import argparse
import pygame
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geosearch import GeosearchEnv
from utils import Utils
from dqn import LunarAgent

# Argument parsing
parser = argparse.ArgumentParser(description="DQN for Geosearch Environment")
parser.add_argument("--verbose", action="store_true", help="Print verbose output")
parser.add_argument("--file_path", type=str, default="./output", help="Directory to save metrics files")
parser.add_argument("--save_metrics", action="store_true", help="Save the metrics of the simulation")
args = parser.parse_args()

# Ensure the directory exists
if args.save_metrics:
    os.makedirs(args.file_path, exist_ok=True)

# Save the training progress plot
if args.save_metrics:
    plt.savefig(os.path.join(args.file_path, 'dqn_training_progress.png'))

# Training environment (no rendering)
env = GeosearchEnv(render_mode=None)
agent = LunarAgent(env)
rewards = agent.train(num_episodes=10, max_steps=365)

# Plot training results
plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
if args.save_metrics:
    plt.savefig(os.path.join(args.file_path, 'dqn_training_progress.png'))
plt.show()
plt.close()

# Function to convert DQN to policy for visualization
def dqn_to_policy(agent, env):
    num_states = env.grid_height * env.grid_width
    policy = np.zeros((num_states, env.action_space.n))
    
    for i in range(env.grid_height):
        for j in range(env.grid_width):
            # Create a sample state
            state = env.reset()[0]  # Use env_display instead of env
            state[2] = (i, j)  # Set position
            
            # Get Q-values from DQN
            with torch.no_grad():
                state_tensor = torch.FloatTensor(agent.process_state(state)).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
            
            # Convert to policy
            best_action = np.argmax(q_values)
            policy[i * env.grid_width + j][best_action] = 1.0
            
    return policy

# Convert DQN to policy for visualization
policy = dqn_to_policy(agent, env)

# Visualize using existing utilities
Utils.render_optimal_policy(
    env,  # Use the display environment instance
    policy,
    save_image=args.save_metrics,
    image_filename=os.path.join(args.file_path, "dqn_policy_visualization.png")
)

# Run the trained agent
Utils.run_optimal_policy(
    env,  # Use the display environment instance
    agent,  # Pass the trained agent
    save_gif=args.save_metrics,
    gif_filename=os.path.join(args.file_path, "dqn_gameplay.gif")
)