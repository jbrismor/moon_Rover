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


# ------------------------------------------
# Environment: GeosearchEnv
# ------------------------------------------
class GeosearchEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(GeosearchEnv, self).__init__()

        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0 

        # Grid setup
        self.grid_height = 20
        self.grid_width = 20
        self.action_space = spaces.Discrete(6)

        # Lunar characteristics  
        self.height_map = Utils.generate_height_map(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            scale=0.1,
            smoothing=2.0,
            seed=37,
            craters=False,
            num_craters=0,
            min_radius=1.0,
            max_radius=3.0,
            min_depth=0.5,
            max_depth=2.0,
            rim_ratio=0.3,
            add_cliffs=False
        )
            
        self.dust_map = Utils.generate_dust_map(self.height_map, self.grid_height, self.grid_width)

        # Time setup
        self.current_day = 0

        # Generate water distribution
        self.water_probability = Utils.generate_water_probability(self.grid_height, self.grid_width)
        self.water_ground_truth = Utils.generate_ground_truth(
            self.water_probability, 
            noise_factor=0.1, 
            threshold=0.1,
            existing_resources=None
        )

        self.confidence_map = np.zeros((self.grid_height, self.grid_width))

        self.gathered_counts = {}
        self.max_gather_times = 20
        self.base_water_reward = 3000
        self.gather_decay = 250

        self.stuck_days = 0
        self.is_stuck = False
        self.last_action = None
        
        self.num_solar_panels = 3
        self.solar_panel_output = 272.2
        self.battery_capacity = 87900
        self.base_consumption = 900
        self.movement_base_energy = 1890
        self.gathering_energy = 1000

        self.agent_pos = None
        self.current_bat_level = self.battery_capacity

        self.max_episode_steps = 365
        self.month_length = 30
        self.last_month_reward = -1

        self.resources_gathered = {
            'water': {'count': 0, 'locations': set()}
        }

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.robot_image = None

        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "water": (15, 94, 156),
            "gold": (255, 207, 64),
            "base_gray": (128, 128, 128),
        }

        self.observation_space = spaces.Dict({
            'ring_heights': spaces.Box(low=0, high=1, shape=(25,), dtype=np.float32),
            'battery': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'position': spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), shape=(2,), dtype=np.float32),
            'sunlight': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'dust': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'water_map': spaces.Box(low=0, high=1, shape=(2, self.grid_height, self.grid_width), dtype=np.float32),
        })

    def _get_local_heights(self, center_i, center_j, size=2):
        patch_size = 2 * size + 1
        local_heights = np.zeros((patch_size, patch_size), dtype=np.float32)
        i_start = max(0, center_i - size)
        i_end = min(self.grid_height, center_i + size + 1)
        j_start = max(0, center_j - size)
        j_end = min(self.grid_width, center_j + size + 1)
        patch_i_start = size - (center_i - i_start)
        patch_j_start = size - (center_j - j_start)
        local_heights[patch_i_start:patch_i_start + (i_end - i_start),
                      patch_j_start:patch_j_start + (j_end - j_start)] = self.height_map[i_start:i_end, j_start:j_end]
        return local_heights.flatten()

    def _local_average_water_prob(self, center, radius=2):
        i, j = center
        i_min = max(0, i - radius)
        i_max = min(self.grid_height, i + radius + 1)
        j_min = max(0, j - radius)
        j_max = min(self.grid_width, j + radius + 1)
        patch = self.water_probability[i_min:i_max, j_min:j_max]
        return np.mean(patch)

    def _gather_potential(self, pos, radius=1):
        i, j = pos
        i_min = max(0, i - radius)
        i_max = min(self.grid_height, i + radius + 1)
        j_min = max(0, j - radius)
        j_max = min(self.grid_width, j + radius + 1)
        patch = self.water_probability[i_min:i_max, j_min:j_max]
        max_prob = np.max(patch)
        potential = max_prob * (self.base_water_reward / 100.0)
        return potential

    def step(self, action):
        i, j = self.agent_pos
        current_local_prob = self._local_average_water_prob(self.agent_pos, radius=2)
        total_reward = 0.0
        next_pos = self.agent_pos
        if (not self.is_stuck) and (self.current_bat_level > 0):
            if action == 0 and i > 0:
                next_pos = (i - 1, j)
            elif action == 1 and i < self.grid_height - 1:
                next_pos = (i + 1, j)
            elif action == 2 and j > 0:
                next_pos = (i, j - 1)
            elif action == 3 and j < self.grid_width - 1:
                next_pos = (i, j + 1)
        if action in [0, 1, 2, 3]:
            next_local_prob = self._local_average_water_prob(next_pos, radius=2)
            if next_local_prob > current_local_prob:
                water_shaping_factor = 10.0
                total_reward += water_shaping_factor * (next_local_prob - current_local_prob)
        terminated, truncated, extra_penalty = self._check_terminal_states(next_pos)
        total_reward += extra_penalty
        if terminated:
            return self._get_observation(), total_reward, terminated, truncated, {}
        current_month = self.current_day // self.month_length
        if current_month > self.last_month_reward:
            total_reward += 10.0
            self.last_month_reward = current_month
        if action == 5 and self.current_bat_level >= self.gathering_energy:
            loc_key = f"{i},{j}"
            gather_count = self.gathered_counts.get(loc_key, 0)
            if self.water_probability[i, j] > 0.65:
                if gather_count == 0:
                    total_reward += 100
                else:
                    total_reward += 0
            else:
                total_reward -= 5
            self.gathered_counts[loc_key] = gather_count + 1
            Utils._update_resource_probabilities(self, i, j)
        if (not self.is_stuck) and (self.current_bat_level > 0):
            self.agent_pos = next_pos
        self.current_day += 1
        sunlight_map = Utils.calculate_sunlight_map(
            self.grid_height, self.grid_width, self.height_map, self.current_day
        )
        next_bat_level, battery_depleted = Utils.calculate_bat(
            (i, j), next_pos, self.current_bat_level,
            self.battery_capacity, sunlight_map,
            self.height_map, self.dust_map,
            action, self.num_solar_panels
        )
        self.current_bat_level = next_bat_level
        if self.current_day >= self.max_episode_steps:
            terminated = True
            truncated = True
        info = {
            'current_day': self.current_day,
            'resources': self.resources_gathered,
            'battery_level': self.current_bat_level,
            'is_stuck': self.is_stuck,
            'is_water_here': bool(self.water_ground_truth[self.agent_pos[0], self.agent_pos[1]])
        }
        self.reward_count += 1
        delta = total_reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = total_reward - self.reward_mean
        self.reward_m2 += delta * delta2
        if self.reward_count > 1:
            reward_std = np.sqrt(self.reward_m2 / (self.reward_count - 1))
        else:
            reward_std = 1.0
        total_reward = (total_reward - self.reward_mean) / reward_std
        return self._get_observation(), total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.dust_map = Utils.generate_dust_map(self.height_map, self.grid_height, self.grid_width)
        self.water_probability = Utils.generate_water_probability(self.grid_height, self.grid_width)
        self.water_ground_truth = Utils.generate_ground_truth(
            self.water_probability, 
            noise_factor=0.1, 
            threshold=0.1,
            existing_resources=None
        )
        self.confidence_map = np.zeros((self.grid_height, self.grid_width))
        self.gathered_counts = {}
        mid_height = self.grid_height // 2
        mid_width = self.grid_width // 2
        min_row = max(0, mid_height - 1)
        max_row = min(self.grid_height - 1, mid_height + 1)
        min_col = max(0, mid_width - 1)
        max_col = min(self.grid_width - 1, mid_width + 1)
        self.agent_pos = (
            random.randint(min_row, max_row),
            random.randint(min_col, max_col),
        )
        self.stuck_days = 0
        self.is_stuck = False
        self.current_bat_level = self.battery_capacity
        self.current_day = 0
        self.last_month_reward = -1
        self.resources_gathered = {
            'water': {'count': 0, 'locations': set()}
        }
        obs = self._get_observation()
        return obs, {}

    def render(self):
        if self.render_mode is None:
            return
        self._init_render()
        self.screen.fill(self.colors["white"])
        font = pygame.font.Font(None, 20)
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_color = self._get_cell_color(i, j)
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
                height_value = self.height_map[i, j]
                height_text = f"{height_value:.1f}"
                text_surface = font.render(height_text, True, self.colors["black"])
                text_rect = text_surface.get_rect(
                    center=(
                        j * self.cell_size + self.cell_size // 2,
                        i * self.cell_size + self.cell_size // 2,
                    )
                )
                self.screen.blit(text_surface, text_rect)
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
        font = pygame.font.Font(None, 36)
        time_text = font.render(f"Lunar Day: {self.current_day}", True, self.colors["white"])
        self.screen.blit(time_text, (10, 10))
        pygame.display.flip()

    def close(self):
        if self.screen is None:
            return
        pygame.quit()
        self.screen = None
        self.clock = None
        self.robot_image = None

    def _check_terminal_states(self, next_pos):
        terminated = False
        truncated = False
        reward_adjustment = 0
        death_prob = Utils.calculate_death_probability(self.current_day)
        if np.random.random() < death_prob:
            terminated = True
            return terminated, truncated, reward_adjustment
        current_dust = Utils.calculate_dust(self.agent_pos, self.dust_map)
        stuck_prob = Utils.calculate_stuck_probability(current_dust)
        if np.random.random() < stuck_prob:
            self.is_stuck = True
            self.stuck_days += 1
            reward_adjustment -= 5
            if self.stuck_days >= 25:
                terminated = True
                reward_adjustment -= 4000
        else:
            self.is_stuck = False
            self.stuck_days = 0
        if not self.is_stuck:
            current_height = Utils.calculate_height(self.agent_pos, self.height_map)
            next_height = Utils.calculate_height(next_pos, self.height_map)
            if Utils.check_crash(current_height, next_height):
                terminated = True
                reward_adjustment -= 4000
        return terminated, truncated, reward_adjustment

    def _init_render(self):
        if self.screen is None and self.render_mode is not None:
            pygame.init()
            self.screen_width = 1225
            self.screen_height = 1225
            self.cell_size = 35
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Lunar Geosearch Environment")
            self.clock = pygame.time.Clock()
            with importlib.resources.path("lunabot", "robot.png") as robot_path:
                self.robot_image = pygame.image.load(str(robot_path))
                self.robot_image = pygame.transform.scale(self.robot_image, (20, 20))

    def _get_cell_color(self, i, j):
        has_water = self.water_ground_truth[i, j]
        dust_level = self.dust_map[i, j]
        sunlight_map = Utils.calculate_sunlight_map(self.grid_height, self.grid_width, self.height_map, self.current_day)
        sunlight_level = Utils.calculate_sunlight_level(sunlight_map, i, j)
        height_level = (self.height_map[i, j] + 50) / 100
        if has_water:
            base_color = np.array(self.colors["water"])
        else:
            dust_factor = 1 - dust_level * 1.5
            base_color = np.array(self.colors["base_gray"]) * dust_factor
        if f"{i},{j}" in self.gathered_counts:
            base_color = np.array([255, 0, 0])
        sunlight_factor = 0.3 + sunlight_level * 0.7
        final_color = np.clip(base_color * sunlight_factor, 0, 255).astype(int)
        return tuple(final_color)

    def _get_observation(self):
        ring_heights = self._get_local_heights(self.agent_pos[0], self.agent_pos[1], size=2)
        ring_heights = (ring_heights + 50) / 100
        battery_level = np.array([self.current_bat_level / self.battery_capacity], dtype=np.float32)
        position = np.array([
            self.agent_pos[0] / (self.grid_height - 1),
            self.agent_pos[1] / (self.grid_width - 1)
        ], dtype=np.float32)
        sunlight_map = Utils.calculate_sunlight_map(self.grid_height, self.grid_width, self.height_map, self.current_day)
        sunlight_level = Utils.calculate_sunlight_level(sunlight_map, self.agent_pos[0], self.agent_pos[1])
        dust = Utils.calculate_dust(self.agent_pos, self.dust_map)
        dust = np.array([dust * 2], dtype=np.float32)
        water_probs = self.water_probability
        agent_mask = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        agent_mask[self.agent_pos[0], self.agent_pos[1]] = 1.0
        water_map = np.stack([water_probs, agent_mask], axis=0).astype(np.float32)
        obs_dict = {
            'ring_heights': ring_heights,
            'battery': battery_level,
            'position': position,
            'sunlight': np.array([sunlight_level], dtype=np.float32),
            'dust': dust,
            'water_map': water_map,
        }
        return obs_dict