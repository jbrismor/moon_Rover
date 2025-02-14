import os
import ray
import pygame
import logging
from datetime import datetime
import numpy as np
from ray.rllib.algorithms.sac import SACConfig
from gymnasium.envs.registration import register
from lunabot.geosearch import GeosearchEnv, Utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)

class RLlibPolicyWrapper:
    """Wrapper for RLlib policies"""
    def __init__(self, algo):
        self.algo = algo
    
    def get_action(self, state):
        return self.algo.compute_single_action(
            observation=state,
            explore=False
        )


def setup_sac_config():
    """Configure SAC algorithm"""
    # Register the environment with Ray
    from ray import tune
    tune.register_env(
        "GeosearchEnv-v0",
        lambda config: GeosearchEnv()
    )

    config = SACConfig()
    config.framework_str = "torch"
    config.env = "GeosearchEnv-v0"
    
    # API configuration
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    
    # Resources
    config.num_workers = 8
    config.evaluation_num_env_runners = 4 
    config.evaluation_interval = 75
    
    # Training parameters
    config.training(
        lr=3e-4,  # Standard learning rate for SAC
        grad_clip=1.0,
        tau=0.005,
        initial_alpha=1.0,
        replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            "capacity": 100000,
        }
    )
    
    # Model configuration
    config.model = {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
        "max_seq_len": 20
    }
    
    # Batch settings
    config.rollout_fragment_length = 1
    config.train_batch_size = 256
    config.target_network_update_freq = 1
    
    return config

def visualize_policy(checkpoint_dir, episodes=5, max_steps=360):
    """Visualize policy behavior"""
    try:
        # Initialize Ray
        ray.init()
        
        # Register environment
        register(
            id="GeosearchEnv-v0",
            entry_point="lunabot.geosearch:GeosearchEnv",
        )
        
        # Create and restore algorithm
        algo = setup_sac_config().build()
        algo.restore(checkpoint_dir)
        
        # Setup environment and policy wrapper
        env = GeosearchEnv(render_mode='human')
        policy_wrapper = RLlibPolicyWrapper(algo)
        
        if env.render_mode != 'human':
            env.render_mode = 'human'
        env._init_render()
        
        frames = []
        total_reward = 0
        
        for episode in range(episodes):
            state, info = env.reset()
            episode_reward = 0
            terminated = False
            steps = 0
            
            while not terminated and steps < max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                env.render()
                frame = pygame.surfarray.array3d(env.screen)
                frames.append(np.transpose(frame, (1, 0, 2)))
                
                action = policy_wrapper.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                pygame.time.wait(66)  # Approximately 15 FPS
                
            total_reward += episode_reward
            logging.info(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
            logging.info(f"Episode {episode + 1} length: {steps} steps")
        
        # Save visualization as GIF
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gif_dir = os.path.join(script_dir, "policy_gifs")
        os.makedirs(gif_dir, exist_ok=True)
        
        gif_path = os.path.join(gif_dir, f"sac_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif")
        Utils.create_gif(frames, filename=gif_path, duration=66)
        logging.info(f"Policy visualization saved to: {gif_path}")
        
        avg_reward = total_reward / episodes
        logging.info(f"Average reward over {episodes} episodes: {avg_reward:.2f}")

    except KeyboardInterrupt:
        logging.info("\nVisualization interrupted by user")
    except Exception as e:
        logging.error(f"Visualization error: {str(e)}")
    finally:
        if 'env' in locals() and hasattr(env, 'close'):
            env.close()
        pygame.quit()
        ray.shutdown()

if __name__ == "__main__":
    # Path to your final checkpoint
    checkpoint_dir = "/Users/jbm/Desktop/moon_Rover/checkpoints/sac_lunar_best"
    
    # Visualize with 5 episodes and 360 max steps
    visualize_policy(checkpoint_dir, episodes=5, max_steps=360)