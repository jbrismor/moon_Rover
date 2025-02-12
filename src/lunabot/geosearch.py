import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os
from .utils import *
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import cm
import importlib.resources


class GeosearchEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(GeosearchEnv, self).__init__()

        # Grid setup
        self.grid_height = 35
        self.grid_width = 35
        self.action_space = spaces.Discrete(6)

        # Lunar characteristics  
        self.height_map = Utils.generate_height_map(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            scale=0.1,
            smoothing=2.0,
            seed=37,
            craters=False,          # Disable crater generation
            num_craters=0,          # (Optional: also set number of craters to 0)
            min_radius=1.0,
            max_radius=3.0,
            min_depth=0.5,
            max_depth=2.0,
            rim_ratio=0.3,
            add_cliffs=False         # Disable cliff formations
        )
            
        # Initial dust map (will be reset each episode)
        self.dust_map = Utils.generate_dust_map(self.height_map, self.grid_height, self.grid_width)

        # Time setup
        self.current_day = 0

        # Generate water and gold distributions
        self.water_probability = Utils.generate_water_probability(self.grid_height, self.grid_width)
        # self.gold_probability = Utils.generate_gold_probability(self.grid_height, self.grid_width)

        # Generate ground truths sequentially to avoid overlap
        self.water_ground_truth = Utils.generate_ground_truth(
            self.water_probability, 
            noise_factor=0.1, 
            threshold=0.1,
            existing_resources=None  # Water goes first
        )

        # Initialize confidence map
        self.confidence_map = np.zeros((self.grid_height, self.grid_width))

        # self.gold_ground_truth = Utils.generate_ground_truth(
        #     self.gold_probability, 
        #     noise_factor=0.2, 
        #     threshold=0.2,
        #     existing_resources=self.water_ground_truth  # Gold avoids water locations
        # )

        # Add new tracking for gathered resources and their decay
        self.gathered_counts = {}  # Dictionary to track number of times each location was gathered
        self.max_gather_times = 20  # Number of times before rewards reach 0 # change from 8 to 20
        self.base_water_reward = 10000  # Base reward for gathering water # change from 200 to 10000
        # self.base_gold_reward = 6000   # Base reward for gathering gold # change from 300 to 600
        self.gather_decay = 250        # Reward decay per gathering

        # Add new state tracking variables
        self.stuck_days = 0  # Track consecutive days stuck
        self.is_stuck = False  # Current stuck status
        self.last_action = None  # Track the last action taken
        
        # Energy system constants
        self.num_solar_panels = 3  # Using 3 panels as suggested
        self.solar_panel_output = 272.2  # Watts per panel
        self.battery_capacity = 87900  # Wh (two batteries) # change from 58600 to 87900 (3 batteries)
        self.base_consumption = 900  # Wh per day # change from 1200 to 900
        self.movement_base_energy = 1890  # Wh per 1000m # changed from 13890 to 1890
        self.gathering_energy = 1000  # Wh # changed from 20000 to 1000

        # Position and battery tracking
        self.agent_pos = None
        self.current_bat_level = self.battery_capacity  # Start with full battery

        # Add timeline constants
        self.max_episode_steps = 365  # One year
        self.month_length = 30  # Days in a month
        self.last_month_reward = -1  # Track last month rewarded

        # Resource tracking
        self.resources_gathered = {
            'water': {'count': 0, 'locations': set()}
            #,'gold': {'count': 0, 'locations': set()}
        }

        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.robot_image = None

        # Colors and background
        # self.background_colors = ["#489030", "#5d924d", "#789030", "#a5803d"]
        # self.background_color_grid = self._generate_background_grid()

        # Colors
        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "water": (15, 94, 156),
            "gold": (255, 207, 64),
            "base_gray": (128, 128, 128),
        }

        # Define proper observation space for continuous values
        self.observation_space = spaces.Dict({
            'ring_heights': spaces.Box(low=-50, high=50, shape=(25,), dtype=np.float32),

            'battery': spaces.Box(low=0, high=self.battery_capacity, shape=(1,), dtype=np.float32),

            'position': spaces.Box(
                low=np.array([0, 0]),
                high=np.array([self.grid_height - 1, self.grid_width - 1]),
                shape=(2,),
                dtype=np.float32
            ),

            'sunlight': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'dust': spaces.Box(low=0, high=0.5, shape=(1,), dtype=np.float32),

            # Flattened water probabilities for the entire map: shape = (35*35=1225,)
            'water_probs': spaces.Box(low=0, high=1, shape=(self.grid_height * self.grid_width,), dtype=np.float32),
            # 'local_probs': spaces.Box(low=0, high=1, shape=(25,), dtype=np.float32),  # 5x5 grid flattened
            # 'local_conf': spaces.Box(low=0, high=1, shape=(25,), dtype=np.float32),   # 5x5 grid flattened
            # 'cardinal_probs': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)  # N,S,E,W averages
        })

    def _get_local_heights(self, center_i, center_j, size=2):
        """
        Returns a 5×5 patch of the height_map centered on (center_i, center_j),
        with proper edge handling. Flattened to shape (25,).
        """
        patch_size = 2 * size + 1  # 5 if size=2
        local_heights = np.zeros((patch_size, patch_size), dtype=np.float32)
        
        # Compute the valid ranges in the main height_map
        i_start = max(0, center_i - size)
        i_end = min(self.grid_height, center_i + size + 1)
        j_start = max(0, center_j - size)
        j_end = min(self.grid_width, center_j + size + 1)
        
        # Compute the offsets in the local patch
        patch_i_start = size - (center_i - i_start)
        patch_j_start = size - (center_j - j_start)
        
        # Slice out the relevant area from self.height_map
        source_slice_i = slice(i_start, i_end)
        source_slice_j = slice(j_start, j_end)
        
        # Slice for the local_heights patch
        target_slice_i = slice(patch_i_start, patch_i_start + (i_end - i_start))
        target_slice_j = slice(patch_j_start, patch_j_start + (j_end - j_start))
        
        # Fill the local_heights patch
        local_heights[target_slice_i, target_slice_j] = self.height_map[source_slice_i, source_slice_j]
        
        return local_heights.flatten()


    # def step(self, action):
    #     """Apply the action and advance time by one hour"""
    #     # Store the action being attempted
    #     self.last_action = action

    #     # Determine base reward based on the action type:
    #     if action in [0, 1, 2, 3]:
    #         # Movement actions (up, down, left, right): reward +1 for taking a step
    #         total_reward = 1
    #     elif action == 4:
    #         # Staying in place
    #         if not self.is_stuck:
    #             total_reward = -5  # Penalty for not moving when not stuck
    #         else:
    #             total_reward = 0   # If stuck, you might not penalize staying
    #     elif action == 5:
    #         # Gather action – start with a base reward of 0;
    #         # any reward will be added below if the gathering is successful.
    #         total_reward = 0

    #     next_pos = self.agent_pos
    #     i, j = self.agent_pos

    #     # Get observation first to ensure it's always defined
    #     obs = self._get_observation()

    #     # Check terminal states before movement
    #     terminated, truncated, reward_adjustment = self._check_terminal_states(next_pos)
    #     total_reward += reward_adjustment

    #     if terminated:
    #         return obs, total_reward, terminated, truncated, {}
        
    #     # Check for month completion reward
    #     current_month = self.current_day // self.month_length
    #     if current_month > self.last_month_reward:
    #         total_reward += 1000  # bonus for surviving another month # change from 100 to 1000
    #         self.last_month_reward = current_month

    #     # movement logic - only execute if not stuck
    #     if not self.is_stuck and self.current_bat_level > 0:
    #         if action == 0 and i > 0:  # Up
    #             next_pos = (i - 1, j)
    #         elif action == 1 and i < self.grid_height - 1:  # Down
    #             next_pos = (i + 1, j)
    #         elif action == 2 and j > 0:  # Left
    #             next_pos = (i, j - 1)
    #         elif action == 3 and j < self.grid_width - 1:  # Right
    #             next_pos = (i, j + 1)
    #         elif action == 4:  # Staying in place
    #             next_pos = (i, j)
    #         elif action == 5:  # Gather
    #             # Only allow gathering if there's enough battery
    #             if self.current_bat_level >= self.gathering_energy:
    #                 # Get location key for tracking
    #                 loc_key = f"{i},{j}"
    #                 gather_count = self.gathered_counts.get(loc_key, 0)
                    
    #                 # Calculate decay factor (linear decay)
    #                 decay_factor = max(0, 1 - (gather_count * self.gather_decay / self.base_water_reward))
    #                 # decay_factor = max(0, 1 - (gather_count * self.gather_decay / 
    #                 #                          (self.base_water_reward if self.water_ground_truth[i, j] else self.base_gold_reward)))
                    
    #                 # Update gather count
    #                 self.gathered_counts[loc_key] = gather_count + 1
                    
    #                 # Apply rewards with decay and track resources
    #                 if self.water_ground_truth[i, j]:
    #                     total_reward += self.base_water_reward * decay_factor
    #                     if loc_key not in self.resources_gathered['water']['locations']:
    #                         self.resources_gathered['water']['count'] += 1
    #                         self.resources_gathered['water']['locations'].add(loc_key)
                    
    #                 # if self.gold_ground_truth[i, j]:
    #                 #     total_reward += self.base_gold_reward * decay_factor
    #                 #     if loc_key not in self.resources_gathered['gold']['locations']:
    #                 #         self.resources_gathered['gold']['count'] += 1
    #                 #         self.resources_gathered['gold']['locations'].add(loc_key)
                    
    #                 # Update probabilities in surrounding area
    #                 Utils._update_resource_probabilities(self, i, j)

    #     # get height and sunlight info for next position
    #     height = Utils.calculate_height(next_pos, self.height_map)
    #     sunlight_map = Utils.calculate_sunlight_map(self.grid_height, self.grid_width, self.height_map, self.current_day)
    #     sunlight_level = Utils.calculate_sunlight_level(sunlight_map, next_pos[0], next_pos[1])

    #     # Calculate new battery level with updated energy system
    #     next_bat_level, battery_depleted = Utils.calculate_bat(
    #         self.agent_pos,
    #         next_pos,
    #         self.current_bat_level,
    #         self.battery_capacity,
    #         sunlight_map,
    #         self.height_map,
    #         self.dust_map,
    #         action,
    #         self.num_solar_panels
    #     )

    #     # Update position if not stuck
    #     if not self.is_stuck and self.current_bat_level > 0:
    #         self.agent_pos = next_pos

    #     # Advance time
    #     self.current_day = self.current_day + 1

    #     # Battery level rewards
    #     if next_bat_level < (0.05 * self.battery_capacity): # change from 0.2 to 0.1 battery level
    #         total_reward -= 10  # Penalty for low battery # change from 20 to 10
    #     elif next_bat_level > (0.95 * self.battery_capacity):
    #         total_reward -= 25  # Penalty for overcharged battery # change from 15 to 25

    #     # Update current battery level
    #     self.current_bat_level = next_bat_level

    #     # Check for episode end due to max steps
    #     if self.current_day >= self.max_episode_steps:
    #         terminated = True
    #         truncated = True

    #     info = {
    #         'current_day': self.current_day,
    #         'resources': self.resources_gathered,
    #         'battery_level': self.current_bat_level,
    #         'is_stuck': self.is_stuck
    #     }

    #     return obs, total_reward, terminated, truncated, info

    # def step(self, action):
    #     """Apply the action and advance time by one hour"""
    #     # Store the action being attempted.
    #     self.last_action = action

    #     # --- Base Reward Setup ---
    #     if action in [0, 1, 2, 3]:
    #         # Movement actions: reward +1 for moving.
    #         total_reward = 1
    #     elif action == 4:
    #         # Staying in place: if not stuck, incur a penalty of -5.
    #         total_reward = -5 if not self.is_stuck else 0
    #     elif action == 5:
    #         # Gathering action: start with 0 reward (will add bonus below).
    #         total_reward = 0

    #     # --- Reward Shaping: Encourage Moving Toward Higher Water Probabilities ---
    #     # Save current water probability at the agent's position.
    #     i, j = self.agent_pos
    #     current_prob = self.water_probability[i, j]

    #     # Determine next position based on the action (if not stuck and if battery > 0).
    #     next_pos = self.agent_pos  # Default: no movement.
    #     if not self.is_stuck and self.current_bat_level > 0:
    #         if action == 0 and i > 0:          # Up
    #             next_pos = (i - 1, j)
    #         elif action == 1 and i < self.grid_height - 1:  # Down
    #             next_pos = (i + 1, j)
    #         elif action == 2 and j > 0:          # Left
    #             next_pos = (i, j - 1)
    #         elif action == 3 and j < self.grid_width - 1:   # Right
    #             next_pos = (i, j + 1)
    #         elif action == 4:  # Stay
    #             next_pos = (i, j)
    #         elif action == 5:
    #             # Gathering action generally does not change position.
    #             next_pos = (i, j)

    #     # For movement actions, compute a potential-based shaping reward.
    #     if action in [0, 1, 2, 3]:
    #         next_prob = self.water_probability[next_pos[0], next_pos[1]]
    #         # The shaping reward is proportional to the difference in water probabilities.
    #         reward_shaping_factor = 5.0  # (Tunable hyperparameter)
    #         shaping_reward = reward_shaping_factor * (next_prob - current_prob)
    #         total_reward += shaping_reward

    #     # --- Terminal Check ---
    #     terminated, truncated, reward_adjustment = self._check_terminal_states(next_pos)
    #     total_reward += reward_adjustment
    #     if terminated:
    #         return self._get_observation(), total_reward, terminated, truncated, {}

    #     # --- Monthly Bonus ---
    #     current_month = self.current_day // self.month_length
    #     if current_month > self.last_month_reward:
    #         total_reward += 1000  # Bonus for surviving an additional month.
    #         self.last_month_reward = current_month

    #     # --- Action-Specific Logic ---
    #     if action == 5:  # Gather action.
    #         if self.current_bat_level >= self.gathering_energy:
    #             loc_key = f"{i},{j}"
    #             gather_count = self.gathered_counts.get(loc_key, 0)
    #             # Compute a linear decay factor for repeated gathering at the same location.
    #             decay_factor = max(0, 1 - (gather_count * self.gather_decay / self.base_water_reward))
    #             self.gathered_counts[loc_key] = gather_count + 1

    #             if self.water_ground_truth[i, j]:
    #                 total_reward += self.base_water_reward * decay_factor
    #                 if loc_key not in self.resources_gathered['water']['locations']:
    #                     self.resources_gathered['water']['count'] += 1
    #                     self.resources_gathered['water']['locations'].add(loc_key)
    #             else:
    #                 # Currently, there is no penalty for gathering in the wrong place.
    #                 # To add a penalty, uncomment the following line:
    #                 total_reward -= 5
    #                 pass

    #     # --- Environment Dynamics ---
    #     # Get current environmental factors for the next position.
    #     height = Utils.calculate_height(next_pos, self.height_map)
    #     sunlight_map = Utils.calculate_sunlight_map(self.grid_height, self.grid_width, self.height_map, self.current_day)
    #     sunlight_level = Utils.calculate_sunlight_level(sunlight_map, next_pos[0], next_pos[1])

    #     # Calculate new battery level and check if battery is depleted.
    #     next_bat_level, battery_depleted = Utils.calculate_bat(
    #         self.agent_pos,
    #         next_pos,
    #         self.current_bat_level,
    #         self.battery_capacity,
    #         sunlight_map,
    #         self.height_map,
    #         self.dust_map,
    #         action,
    #         self.num_solar_panels
    #     )

    #     # Update the agent's position if not stuck and if there is remaining battery.
    #     if not self.is_stuck and self.current_bat_level > 0:
    #         self.agent_pos = next_pos

    #     # Advance time (e.g., one day or one time unit).
    #     self.current_day += 1

    #     # --- Battery Level Penalties ---
    #     if next_bat_level < (0.1 * self.battery_capacity):
    #         total_reward -= 20  # Penalty for critically low battery.
    #     elif next_bat_level > (0.95 * self.battery_capacity):
    #         total_reward -= 15  # Penalty for overcharged battery conditions.

    #     self.current_bat_level = next_bat_level

    #     # --- Episode Termination Check ---
    #     if self.current_day >= self.max_episode_steps:
    #         terminated = True
    #         truncated = True

    #     info = {
    #         'current_day': self.current_day,
    #         'resources': self.resources_gathered,
    #         'battery_level': self.current_bat_level,
    #         'is_stuck': self.is_stuck
    #     }

    #     return self._get_observation(), total_reward, terminated, truncated, info

    ##############################################################################################################
    ############################# NEW FUNCTIONS ##################################################################
    ##############################################################################################################

    def _local_average_water_prob(self, center, radius=2):
        """
        Returns the average water probability in a local patch around `center`.
        The patch is (2*radius+1) x (2*radius+1).
        """
        i, j = center
        i_min = max(0, i - radius)
        i_max = min(self.grid_height, i + radius + 1)
        j_min = max(0, j - radius)
        j_max = min(self.grid_width, j + radius + 1)

        patch = self.water_probability[i_min:i_max, j_min:j_max]
        return np.mean(patch)

    def _gather_potential(self, pos, radius=1):
        """
        Returns a 'potential' value for how good it is to gather near `pos`.
        For simplicity, we use the max water probability in a local patch.
        Then scale that by some fraction of the base water reward so the maximum
        potential won't exceed a fraction of the actual gather reward.
        """
        i, j = pos
        i_min = max(0, i - radius)
        i_max = min(self.grid_height, i + radius + 1)
        j_min = max(0, j - radius)
        j_max = min(self.grid_width, j + radius + 1)

        patch = self.water_probability[i_min:i_max, j_min:j_max]
        max_prob = np.max(patch)  # highest water prob near pos

        # For example, scale by (base_water_reward / 100).
        # If base_water_reward=5000, potential could be up to 50 if max_prob=1.
        potential = max_prob * (self.base_water_reward / 100.0)
        return potential

    def step(self, action):
        """
        Apply the action (0=Up, 1=Down, 2=Left, 3=Right, 4=Stay, 5=Gather)
        and advance time by one day (or hour) in the environment.
        """
        # Record the chosen action for rendering/analysis.
        self.last_action = action

        # -------------------
        # 1) Base Action Rewards
        # -------------------
        if action in [0, 1, 2, 3]:
            # Movement actions
            total_reward = 1   # +1 reward for moving
        elif action == 4:
            # Stay in place
            total_reward = -5 if not self.is_stuck else 0
        elif action == 5:
            # Gather action
            total_reward = 0   # We'll add gather reward if water is found
        else:
            # Should never happen in a Discrete(6) environment
            total_reward = 0

        # -------------------
        # 2) Prepare for Shaping
        # -------------------
        current_local_prob = self._local_average_water_prob(self.agent_pos, radius=2)
        current_phi = self._gather_potential(self.agent_pos, radius=1)

        i, j = self.agent_pos
        next_pos = self.agent_pos  # Default to same position.

        # -------------------
        # 3) Movement/Next Pos Logic
        # -------------------
        if not self.is_stuck and self.current_bat_level > 0:
            if action == 0 and i > 0:             # Up
                next_pos = (i - 1, j)
            elif action == 1 and i < self.grid_height - 1:  # Down
                next_pos = (i + 1, j)
            elif action == 2 and j > 0:           # Left
                next_pos = (i, j - 1)
            elif action == 3 and j < self.grid_width - 1:   # Right
                next_pos = (i, j + 1)
            # action == 4 => Stay put
            # action == 5 => Gather does not move agent

        # -------------------
        # 4) Reward Shaping for Movement
        # -------------------
        if action in [0, 1, 2, 3]:
            # Compare local water probs around current vs. next pos.
            next_local_prob = self._local_average_water_prob(next_pos, radius=2)
            shaping_factor = 3.0  # tune as needed
            shaping_reward = shaping_factor * (next_local_prob - current_local_prob)
            total_reward += shaping_reward

            # Potential-based shaping for gathering
            next_phi = self._gather_potential(next_pos, radius=1)
            potential_diff = next_phi - current_phi
            total_reward += potential_diff

        # -------------------
        # 5) Check Terminal States (before gather)
        # -------------------
        terminated, truncated, reward_adjustment = self._check_terminal_states(next_pos)
        total_reward += reward_adjustment
        if terminated:
            return self._get_observation(), total_reward, terminated, truncated, {}

        # -------------------
        # 6) Monthly Bonus
        # -------------------
        current_month = self.current_day // self.month_length
        if current_month > self.last_month_reward:
            total_reward += 1000
            self.last_month_reward = current_month

        # -------------------
        # 7) Gather Logic
        # -------------------
        if action == 5 and self.current_bat_level >= self.gathering_energy:
            loc_key = f"{i},{j}"
            gather_count = self.gathered_counts.get(loc_key, 0)

            # Decay factor for repeated gathers in the same spot (when there's water).
            decay_factor = max(0, 1 - (gather_count * self.gather_decay / self.base_water_reward))

            # Update gather count
            self.gathered_counts[loc_key] = gather_count + 1

            if self.water_ground_truth[i, j]:
                # There's water => big reward
                total_reward += self.base_water_reward * decay_factor

                # Track resource if not already done
                if loc_key not in self.resources_gathered['water']['locations']:
                    self.resources_gathered['water']['count'] += 1
                    self.resources_gathered['water']['locations'].add(loc_key)
            else:
                # NO WATER => new logic:
                if gather_count == 0:
                    # First time gathering in this waterless cell => small reward
                    total_reward += 1
                else:
                    # Subsequent gathers => small penalty
                    total_reward -= 1

            # Update environment's water/confidence maps
            Utils._update_resource_probabilities(self, i, j)

        # -------------------
        # 8) Environment Dynamics
        # -------------------
        if not self.is_stuck and self.current_bat_level > 0:
            self.agent_pos = next_pos

        self.current_day += 1

        from_pos = (i, j)
        to_pos = next_pos
        sunlight_map = Utils.calculate_sunlight_map(
            self.grid_height, self.grid_width, self.height_map, self.current_day
        )
        next_bat_level, battery_depleted = Utils.calculate_bat(
            from_pos,
            to_pos,
            self.current_bat_level,
            self.battery_capacity,
            sunlight_map,
            self.height_map,
            self.dust_map,
            action,
            self.num_solar_panels
        )
        self.current_bat_level = next_bat_level

        # Battery-based penalties
        if next_bat_level < (0.1 * self.battery_capacity):
            total_reward -= 20
        elif next_bat_level > (0.95 * self.battery_capacity):
            total_reward -= 15

        # -------------------
        # 9) Episode End Check
        # -------------------
        if self.current_day >= self.max_episode_steps:
            terminated = True
            truncated = True

        info = {
            'current_day': self.current_day,
            'resources': self.resources_gathered,
            'battery_level': self.current_bat_level,
            'is_stuck': self.is_stuck
        }

        return self._get_observation(), total_reward, terminated, truncated, info



    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        """Reset the environment with random starting position and midnight time"""
        # Generate new dust distribution
        self.dust_map = Utils.generate_dust_map(self.height_map, self.grid_height, self.grid_width)

        # Generate water and gold distributions
        self.water_probability = Utils.generate_water_probability(self.grid_height, self.grid_width)
        # self.gold_probability = Utils.generate_gold_probability(self.grid_height, self.grid_width)

        # Generate ground truths sequentially to avoid overlap
        self.water_ground_truth = Utils.generate_ground_truth(
            self.water_probability, 
            noise_factor=0.1, 
            threshold=0.1,
            existing_resources=None  # Water goes first
        )

        # Reset confidence map
        self.confidence_map = np.zeros((self.grid_height, self.grid_width))

        # self.gold_ground_truth = Utils.generate_ground_truth(
        #     self.gold_probability, 
        #     noise_factor=0.2, 
        #     threshold=0.2,
        #     existing_resources=self.water_ground_truth  # Gold avoids water locations
        # )

        # Clear gathered resources tracking
        self.gathered_counts = {}

        # find the middle of the grid
        mid_height = self.grid_height // 2
        mid_width = self.grid_width // 2

        # Calculate bounds for a 5x5 square centered around the middle
        min_row = max(0, mid_height - 1)
        max_row = min(self.grid_height - 1, mid_height + 1)
        min_col = max(0, mid_width - 1)
        max_col = min(self.grid_width - 1, mid_width + 1)

        # Reset position and time
        self.agent_pos = (
            random.randint(min_row, max_row),
            random.randint(min_col, max_col),
        )
        # height = Utils.calculate_height(self.agent_pos, self.height_map)

        # # Precompute sunlight map and extract sunlight level for the starting position
        # sunlight_map = Utils.calculate_sunlight_map(self.grid_height, self.grid_width, self.height_map, self.current_day)
        # sunlight_level = Utils.calculate_sunlight_level(
        #     sunlight_map, self.agent_pos[0], self.agent_pos[1]
        # )

        # dust = Utils.calculate_dust(self.agent_pos, self.dust_map)

        # Reset terminal state tracking
        self.stuck_days = 0
        self.is_stuck = False
        self.current_bat_level = self.battery_capacity  # Start with full battery

        # Reset timeline tracking
        self.current_day = 0
        self.last_month_reward = -1
        
        # Reset resource tracking
        self.resources_gathered = {
            'water': {'count': 0, 'locations': set()}
            #,'gold': {'count': 0, 'locations': set()}
        }

        obs = self._get_observation()
        return obs, {}

    def render(self):
        if self.render_mode is None:
            return

        self._init_render()

        self.screen.fill(self.colors["white"])

        def get_height_text_color(height, cell_color):
            # Get base color from terrain colormap
            normalized = (height + 50) / 100
            terrain_color = cm.terrain(normalized)
            base_text_color = tuple(int(c * 255) for c in terrain_color[:3])

            # Calculate cell brightness
            cell_brightness = sum(cell_color) / 3

            # Adjust text color intensity based on cell brightness
            # For dark backgrounds, make text brighter
            if cell_brightness < 128:
                adjusted_color = tuple(min(255, c + 40) for c in base_text_color)
            # For light backgrounds, make text darker
            else:
                adjusted_color = tuple(max(0, c - 40) for c in base_text_color)

            # Apply cell brightness factor to maintain readability
            brightness_factor = max(
                0.3, cell_brightness / 255
            )  # minimum brightness of 0.3
            final_color = tuple(int(c * brightness_factor) for c in adjusted_color)

            return final_color

        # Draw grid cells with characteristics
        font = pygame.font.Font(None, 20)

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Use original cell coloring
                cell_color = self._get_cell_color(i, j)

                # Draw the base cell
                pygame.draw.rect(
                    self.screen,
                    cell_color,
                    pygame.Rect(
                        j * self.cell_size,
                        i * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

                # Draw grid lines
                pygame.draw.rect(
                    self.screen,
                    self.colors["black"],
                    pygame.Rect(
                        j * self.cell_size,
                        i * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                    1,
                )

                # Get actual height value
                height_value = self.height_map[i, j]

                # Get text color based on height and cell color
                text_color = get_height_text_color(height_value, cell_color)

                # Draw height number with colored text
                height_text = f"{height_value:.1f}"
                text_surface = font.render(height_text, True, text_color)
                text_rect = text_surface.get_rect(
                    center=(
                        j * self.cell_size + self.cell_size // 2,
                        i * self.cell_size + self.cell_size // 2,
                    )
                )
                self.screen.blit(text_surface, text_rect)

        # Draw the rover with safe size calculation
        height_at_pos = self.height_map[self.agent_pos]
        base_size = 25
        size_factor = max(0.2, min(2.0, 0.4 + ((height_at_pos + 50) / 100) * 1.2))
        adjusted_size = max(10, int(base_size * size_factor))

        scaled_robot = pygame.transform.scale(
            self.robot_image, (adjusted_size, adjusted_size)
        )
        self.screen.blit(
            scaled_robot,
            (
                self.agent_pos[1] * self.cell_size
                + (self.cell_size - adjusted_size) // 2,
                self.agent_pos[0] * self.cell_size
                + (self.cell_size - adjusted_size) // 2,
            ),
        )

        # Display info with actual height value from agent position
        font = pygame.font.Font(None, 36)
        time_text = font.render(f"Lunar Day: {self.current_day}", True, self.colors["white"])
        height_text = font.render(f"Height: {self.height_map[self.agent_pos]:.1f}m", True, self.colors["white"])
        dust_text = font.render(f"Dust: {self.dust_map[self.agent_pos]:.2f}m", True, self.colors["white"])

        # Add new battery percentage display
        battery_percentage = (self.current_bat_level / self.battery_capacity) * 100
        battery_text = font.render(f"Battery: {battery_percentage:.1f}%", True, self.colors["white"])

        # Add stuck status display
        stuck_text = font.render(f"Stuck: {str(self.is_stuck)}", True, self.colors["white"])

        # Add last action display
        action_names = {
            0: "Up",
            1: "Down",
            2: "Left",
            3: "Right",
            4: "Stay",
            5: "Gather"
        }
        last_action = getattr(self, 'last_action', None)  # Get last_action if it exists, otherwise None
        action_text = font.render(f"Action: {action_names.get(last_action, 'None')}", True, self.colors["white"])

        # Add resource tracking display
        water_count = self.resources_gathered['water']['count']
        # gold_count = self.resources_gathered['gold']['count']
        water_text = font.render(f"Water: {water_count}", True, self.colors["white"])
        # gold_text = font.render(f"Gold: {gold_count}", True, self.colors["white"])

        # Display all text on the screen
        self.screen.blit(time_text, (10, 10))
        self.screen.blit(height_text, (10, 50))
        self.screen.blit(dust_text, (10, 90))
        self.screen.blit(battery_text, (10, 130))
        self.screen.blit(stuck_text, (10, 170))
        self.screen.blit(action_text, (10, 210))
        self.screen.blit(water_text, (10, 250))
        # self.screen.blit(gold_text, (10, 290))

        pygame.display.flip()

    def close(self):
        if self.screen is None:
            return
        pygame.quit()
        self.screen = None
        self.clock = None
        self.robot_image = None

    def _check_terminal_states(self, next_pos):
        """
        Check all terminal state conditions.
        Returns: (terminated, truncated, reward_adjustment)
        """
        terminated = False
        truncated = False
        reward_adjustment = 0
        
        # 1. Check for random death
        death_prob = Utils.calculate_death_probability(self.current_day)
        if np.random.random() < death_prob:
            terminated = True
            return terminated, truncated, reward_adjustment
        
        # 2. Check for stuck state
        current_dust = Utils.calculate_dust(self.agent_pos, self.dust_map)
        stuck_prob = Utils.calculate_stuck_probability(current_dust)
        
        # Determine if agent becomes/remains stuck
        if np.random.random() < stuck_prob:
            self.is_stuck = True
            self.stuck_days += 1
            reward_adjustment -= 5  # dailyu penalty for being stuck # change from 30 to 20
            if self.stuck_days >= 25:  # Terminal state after 5 days stuck # changed to 25
                terminated = True
                reward_adjustment -= 10000  # additional terminal p[enalty for being stuck too long# change from 100000 to 10000
        else:
            self.is_stuck = False
            self.stuck_days = 0
        
        # 3. Check for crash
        if not self.is_stuck:  # Only check for crash if not stuck
            current_height = Utils.calculate_height(self.agent_pos, self.height_map)
            next_height = Utils.calculate_height(next_pos, self.height_map)
            if Utils.check_crash(current_height, next_height):
                terminated = True
                reward_adjustment -= 10000  # Crash penalty # changed from 100000 to 10000
        
        return terminated, truncated, reward_adjustment

    def _init_render(self):
        """Initialize Pygame components if not already initialized"""
        if self.screen is None and self.render_mode is not None:
            pygame.init()
            self.screen_width = 1225  # Screen dimensions
            self.screen_height = 1225
            self.cell_size = 35  # Size of each grid cell

            # Create Pygame window and set the title
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Lunar Geosearch Environment")
            self.clock = pygame.time.Clock()

            # Use importlib.resources to load the robot image
            with importlib.resources.path("lunabot", "robot.png") as robot_path:
                self.robot_image = pygame.image.load(str(robot_path))
                self.robot_image = pygame.transform.scale(self.robot_image, (20, 20))

    def _get_cell_color(self, i, j):
        """Determine the color of a cell based on resources, dust, sunlight, and height"""
        # Check ground truth for resources
        has_water = self.water_ground_truth[i, j]
        # has_gold = self.gold_ground_truth[i, j]

        # Get other environmental factors
        dust_level = self.dust_map[i, j]
        sunlight_map = Utils.calculate_sunlight_map(self.grid_height, self.grid_width, self.height_map, self.current_day)
        sunlight_level = Utils.calculate_sunlight_level(sunlight_map, i, j)
        height_level = (self.height_map[i, j] + 50) / 100  # Normalize height from -50 to 50 to 0-1

        # Determine base color from resources
        if has_water:
            base_color = np.array(self.colors["water"])
        # elif has_gold:
        #     base_color = np.array(self.colors["gold"])
        else:
            # Base gray color modified by dust
            dust_factor = 1 - dust_level * 1.5
            base_color = np.array(self.colors["base_gray"]) * dust_factor

        # Apply gathering visualization
        if f"{i},{j}" in self.gathered_counts:
            base_color = np.array([255, 0, 0])  # Red dot for gathered locations

        # Finally always apply sunlight effect
        sunlight_factor = 0.3 + sunlight_level * 0.7  # 0.3 is minimum brightness
        final_color = np.clip(base_color * sunlight_factor, 0, 255).astype(int)

        return tuple(final_color)

    def _is_resource(self, position, resource_type=None, threshold=0.05):
        """
        Determines if a given position contains a resource (water or gold).
        The threshold parameter defines the minimum probability value to consider a resource present.
        """
        i, j = position

        # returns true if resource is present
        if resource_type == "water":
            return self.water_ground_truth[i, j] > threshold
        # elif resource_type == "gold":
        #     return self.gold_ground_truth[i, j] > threshold
        # else:
        #     return self.water_ground_truth[i, j] or self.gold_ground_truth[i, j]
        else:
            return self.water_ground_truth[i, j]  # Simplified for water-only

    # def _get_local_view(self, center_i, center_j, size=2):
    #     """
    #     Get local view with proper edge handling.
    #     """
    #     # Create padded arrays to handle edge cases
    #     padded_probs = np.zeros((2*size + 1, 2*size + 1), dtype=np.float32)
    #     padded_conf = np.zeros((2*size + 1, 2*size + 1), dtype=np.float32)
        
    #     # Calculate valid ranges for both source and target arrays
    #     i_start_source = max(0, center_i - size)
    #     i_end_source = min(self.grid_height, center_i + size + 1)
    #     j_start_source = max(0, center_j - size)
    #     j_end_source = min(self.grid_width, center_j + size + 1)
        
    #     i_start_target = size - (center_i - i_start_source)
    #     j_start_target = size - (center_j - j_start_source)
        
    #     # Copy valid data from environment to padded arrays
    #     source_slice_i = slice(i_start_source, i_end_source)
    #     source_slice_j = slice(j_start_source, j_end_source)
    #     target_slice_i = slice(i_start_target, i_start_target + (i_end_source - i_start_source))
    #     target_slice_j = slice(j_start_target, j_start_target + (j_end_source - j_start_source))
        
    #     padded_probs[target_slice_i, target_slice_j] = self.water_probability[source_slice_i, source_slice_j]
    #     padded_conf[target_slice_i, target_slice_j] = self.confidence_map[source_slice_i, source_slice_j]
        
    #     return padded_probs.flatten(), padded_conf.flatten()
    
    # def _get_cardinal_averages(self, center_i, center_j, local_size=2):
    #     """
    #     Optimized calculation of average probabilities in each cardinal direction.
    #     """
    #     # Initialize counters and sums
    #     directions = {
    #         'north': {'sum': 0.0, 'count': 0},
    #         'south': {'sum': 0.0, 'count': 0},
    #         'east': {'sum': 0.0, 'count': 0},
    #         'west': {'sum': 0.0, 'count': 0}
    #     }
        
    #     # Calculate local view bounds to exclude
    #     local_min_i = max(0, center_i - local_size)
    #     local_max_i = min(self.grid_height - 1, center_i + local_size)
    #     local_min_j = max(0, center_j - local_size)
    #     local_max_j = min(self.grid_width - 1, center_j + local_size)
        
    #     # Process rows above center (north)
    #     if center_i > 0:
    #         north_slice = self.water_probability[0:local_min_i, :]
    #         directions['north']['sum'] = np.sum(north_slice)
    #         directions['north']['count'] = north_slice.size
        
    #     # Process rows below center (south)
    #     if center_i < self.grid_height - 1:
    #         south_slice = self.water_probability[local_max_i+1:, :]
    #         directions['south']['sum'] = np.sum(south_slice)
    #         directions['south']['count'] = south_slice.size
        
    #     # Process columns left of center (west)
    #     if center_j > 0:
    #         west_slice = self.water_probability[:, 0:local_min_j]
    #         directions['west']['sum'] = np.sum(west_slice)
    #         directions['west']['count'] = west_slice.size
        
    #     # Process columns right of center (east)
    #     if center_j < self.grid_width - 1:
    #         east_slice = self.water_probability[:, local_max_j+1:]
    #         directions['east']['sum'] = np.sum(east_slice)
    #         directions['east']['count'] = east_slice.size
        
    #     # Calculate averages
    #     averages = np.zeros(4, dtype=np.float32)
    #     for idx, direction in enumerate(['north', 'south', 'east', 'west']):
    #         if directions[direction]['count'] > 0:
    #             averages[idx] = directions[direction]['sum'] / directions[direction]['count']
        
    #     return averages
    
    def _get_observation(self):
        """Return the current observation as a dictionary matching self.observation_space."""
        # Basic scalar observations
        height = Utils.calculate_height(self.agent_pos, self.height_map)
        sunlight_map = Utils.calculate_sunlight_map(
            self.grid_height, self.grid_width, self.height_map, self.current_day
        )
        sunlight_level = Utils.calculate_sunlight_level(
            sunlight_map, self.agent_pos[0], self.agent_pos[1]
        )
        dust = Utils.calculate_dust(self.agent_pos, self.dust_map)

        # New ring heights (5×5) around the agent
        ring_heights = self._get_local_heights(self.agent_pos[0], self.agent_pos[1], size=2)

        # Full water probability map, flattened
        water_probs = self.water_probability.flatten()

        obs_dict = {
            'ring_heights': ring_heights,  # shape (25,)

            'battery': np.array([self.current_bat_level], dtype=np.float32),
            'position': np.array(self.agent_pos, dtype=np.float32),
            'sunlight': np.array([sunlight_level], dtype=np.float32),
            'dust': np.array([dust], dtype=np.float32),

            'water_probs': water_probs,    # shape (35*35 = 1225,)
        }

        return obs_dict