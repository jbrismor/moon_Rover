import argparse
import pygame
import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geosearch import GeosearchEnv  # Import Geosearch environment
from solvers import DynamicProgramming #, MonteCarlo, TemporalDifference  # Import solvers
from utils import Utils  # Import the Utils class

# Argument parsing
parser = argparse.ArgumentParser(description="RL Solvers for Geosearch Environments")
parser.add_argument(
    "--env",
    choices=["geosearch"],
    default="geosearch",
    help="Specify the environment to solve: geosearch",
)
parser.add_argument(
    "--algo",
    # choices=["dp", "mc", "td"],
    choices=["dp"],
    default="dp",
    help="Specify the algorithm to use: dynamic programming (dp), Monte Carlo (mc), or Temporal Difference (td)",
)
parser.add_argument("--verbose", action="store_true", help="Print verbose output")
parser.add_argument(
    "--file_path",
    type=str,
    default="./output",
    help="Directory to save metrics files",
)
parser.add_argument(
    "--save_metrics",
    action="store_true",
    help="Save the metrics of the simulation",
)
parser.add_argument(
    "--show_training",
    action="store_true",
    help="Display training visualization while the agent trains",
)
args = parser.parse_args()

# Set environment
env = GeosearchEnv(render_mode="human" if args.show_training else None)

# Set algorithm parameters
# if args.algo == "dp":
#     episodes = 5000
#     max_steps = 100
#     solver = DynamicProgramming(env)
# elif args.algo == "mc":
#     episodes = 5000
#     max_steps = 50
#     solver = MonteCarlo(env)
# else:  # td
#     episodes = 5000
#     max_steps = 37
#     solver = TemporalDifference(env)

episodes = 500
max_steps = 25
solver = DynamicProgramming(env)

# Define output file paths
image_filename = os.path.join(args.file_path, "policy_visualization.png")
gif_filename = os.path.join(args.file_path, "gameplay.gif")
convergence_plot_filename = os.path.join(args.file_path, "convergence_plot.png")

# Train the agent
policy = solver.train(max_steps=max_steps, episodes=episodes, verbose=args.verbose)

# Display the policy
print("\nOptimal Policy:")
print(policy)

# Reinitialize environment with render_mode="human" for rendering the optimal policy
env = GeosearchEnv(render_mode="human")

# Render the optimal policy
Utils.render_optimal_policy(
    env, policy, save_image=args.save_metrics, image_filename=image_filename
)

# Wait for a key press before continuing visualization of the agent running the optimal policy
# print(
#     "Press any key while in pygame window to continue to visualize the agent running the optimal policy..."
# )
# waiting = True
# while waiting:
#     for event in pygame.event.get():
#         if event.type == pygame.KEYDOWN:
#             waiting = False
#             break
#         elif event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()

# Run the environment again with the optimal policy and render it
Utils.run_optimal_policy(
    env, policy, save_gif=args.save_metrics, gif_filename=gif_filename
)

# Plot convergence if saving metrics
if args.save_metrics:
    Utils.plot_convergence(solver.mean_reward, file_path=convergence_plot_filename)

