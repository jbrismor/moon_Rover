import matplotlib.pyplot as plt
import pickle
import pygame
import noise
import os
import numpy as np
from PIL import Image  # Add for GIF creation
from scipy.ndimage import gaussian_filter 
import torch


class Utils:
    @staticmethod
    def _ensure_directory_exists(file_path):
        """Ensures the parent directory of the given file path exists."""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def save_policy(policy, filename="optimal_policy.pkl"):
        """Saves the optimal policy to a file."""
        Utils._ensure_directory_exists(filename)
        with open(filename, "wb") as f:
            pickle.dump(policy, f)
        print(f"Optimal policy saved to {filename}")

    @staticmethod
    def load_policy(filename="optimal_policy.pkl"):
        """Loads the optimal policy from a file."""
        Utils._ensure_directory_exists(filename)
        with open(filename, "rb") as f:
            policy = pickle.load(f)
        print(f"Optimal policy loaded from {filename}")
        return policy

    @staticmethod
    def _state_to_index(env, state):
        """Converts a (row, col) tuple into a unique index for geosearch or returns the state for discrete spaces."""
        return state[0] * env.grid_width + state[1]

    @staticmethod
    def generate_dust_map(height_map, grid_height: int=25, grid_width: int=25, scale=0.2, smoothing=2.0, dust_height_correlation=0.25):
        """
        Generate a dust distribution map based on height and random noise.
        
        Args:
            height_map (np.ndarray): The terrain height map
            scale (float): Scale of the noise variation (default: 0.1)
            smoothing (float): Gaussian smoothing sigma value (default: 2.0)
            dust_height_correlation (float): How strongly dust correlates with height (0-1)
        
        Returns:
            np.ndarray: Dust distribution map with values between 0-1
        """
        grid_height, grid_width = height_map.shape
        
        # Create base random noise
        base_dust = np.random.normal(0, scale, (grid_height, grid_width))
        
        # Smooth the noise
        dust_map = gaussian_filter(base_dust, sigma=smoothing)
        
        # Normalize dust to 0-1
        dust_map = (dust_map - dust_map.min()) / (dust_map.max() - dust_map.min())
        
        # Create height influence (normalize height to 0-1 first and invert it)
        height_normalized = (height_map - height_map.min()) / (height_map.max() - height_map.min())
        height_influence = 1 - height_normalized  # Invert so lower areas have more dust
        
        # Combine dust and height influence
        final_dust = (dust_height_correlation * height_influence + 
                    (1 - dust_height_correlation) * dust_map)
        
        # Final normalization to 0-1
        final_dust = (final_dust - final_dust.min()) / (final_dust.max() - final_dust.min())

        # Scale to 0-0.5 range
        final_dust = final_dust * 0.5
        
        return final_dust

    @staticmethod
    def generate_height_map(grid_height: int=35,
                            grid_width: int=35,
                            scale=0.1,
                            smoothing=2.0,
                            seed=None,
                            min_height: int=-50,
                            max_height: int=50,
                            craters=True,
                            num_craters=3,
                            min_radius=1.0,
                            max_radius=3.0,
                            min_depth=0.5,
                            max_depth=2.0,
                            rim_ratio=0.3,
                            add_cliffs=True):
        """
        Generate terrain with optional craters and natural cliff formations along mountain ranges. 
        Uses the same seed for all random operations to ensure reproducibility.
        
        New Parameters:
            add_cliffs (bool): Whether to add cliff formations along mountain ranges
        """
        # Store the current random state
        if seed is not None:
            original_state = np.random.get_state()
            np.random.seed(seed)
        
        try:
            # Create base terrain
            noise = np.random.normal(0, scale, (grid_height, grid_width))
            surface = gaussian_filter(noise, sigma=smoothing)
            surface = gaussian_filter(surface, sigma=smoothing/2)    
            
            # Scale to initial range (temporary)
            surface = (surface - surface.min()) / (surface.max() - surface.min())
            
            if add_cliffs:
                # Generate mountain range paths using random walks
                num_ranges = 2  # Number of mountain ranges with cliffs
                for _ in range(num_ranges):
                    # Start point for mountain range
                    x = np.random.randint(5, grid_width-5)
                    y = np.random.randint(5, grid_height-5)
                    
                    # Generate mountain range path
                    path_length = np.random.randint(10, 20)
                    mountain_path = []
                    current_x, current_y = x, y
                    
                    # Random walk with momentum
                    dx = np.random.choice([-1, 0, 1])
                    dy = np.random.choice([-1, 0, 1])
                    
                    for _ in range(path_length):
                        mountain_path.append((current_x, current_y))
                        
                        # Update direction with momentum (70% chance to keep direction)
                        if np.random.random() > 0.3:
                            new_dx = dx + np.random.choice([-1, 0, 1])
                            new_dy = dy + np.random.choice([-1, 0, 1])
                            dx = np.clip(new_dx, -1, 1)
                            dy = np.clip(new_dy, -1, 1)
                        
                        # Update position
                        current_x = np.clip(current_x + dx, 5, grid_width-6)
                        current_y = np.clip(current_y + dy, 5, grid_height-6)
                    
                    # Create elevation along the mountain range
                    for px, py in mountain_path:
                        # Create mountain peak
                        peak_height = np.random.uniform(0.7, 1.0)
                        
                        # Create cliff on one side
                        cliff_direction = np.random.choice(['N', 'S', 'E', 'W'])
                        cliff_width = np.random.randint(3, 6)
                        
                        # Apply elevation to a local area
                        for i in range(-2, 3):
                            for j in range(-2, 3):
                                if 0 <= py+i < grid_height and 0 <= px+j < grid_width:
                                    dist = np.sqrt(i**2 + j**2)
                                    height_factor = peak_height * np.exp(-dist/2)
                                    
                                    # Apply cliff effect
                                    if cliff_direction == 'N' and i > 0:
                                        height_factor *= np.exp(-i/cliff_width)
                                    elif cliff_direction == 'S' and i < 0:
                                        height_factor *= np.exp(i/cliff_width)
                                    elif cliff_direction == 'E' and j < 0:
                                        height_factor *= np.exp(j/cliff_width)
                                    elif cliff_direction == 'W' and j > 0:
                                        height_factor *= np.exp(-j/cliff_width)
                                    
                                    surface[py+i, px+j] = max(surface[py+i, px+j], height_factor)
            
            # Scale to final height range
            surface = (surface - surface.min()) / (surface.max() - surface.min())
            surface = surface * (max_height - min_height) + min_height
            
            # Add craters if specified
            if craters:
                # Define the landing area boundaries (middle 5x5)
                landing_min_x = grid_width//2 - 2
                landing_max_x = grid_width//2 + 2
                landing_min_y = grid_height//2 - 2
                landing_max_y = grid_height//2 + 2
                
                for _ in range(num_craters):
                    while True:
                        center_x = np.random.randint(0, grid_width)
                        center_y = np.random.randint(0, grid_height)
                        radius = np.random.uniform(min_radius, max_radius)
                        
                        crater_too_close = False
                        safe_distance = radius * 1.2
                        
                        if (center_x + safe_distance >= landing_min_x and 
                            center_x - safe_distance <= landing_max_x and 
                            center_y + safe_distance >= landing_min_y and 
                            center_y - safe_distance <= landing_max_y):
                            crater_too_close = True
                        
                        if not crater_too_close:
                            break
                    
                    depth = np.random.uniform(min_depth, max_depth)
                    rim_height = depth * rim_ratio
                    
                    y, x = np.mgrid[0:grid_height, 0:grid_width]
                    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    crater_mask = distances <= radius
                    rim_mask = (distances > radius * 0.8) & (distances <= radius * 1.2)
                    
                    surface[crater_mask] -= depth * np.cos(np.pi * distances[crater_mask] / (2 * radius))
                    rim_distances = (distances[rim_mask] - radius * 0.8) / (radius * 0.4)
                    surface[rim_mask] += rim_height * np.sin(np.pi * rim_distances)
            
            # Final smoothing to blend everything together
            surface = gaussian_filter(surface, sigma=0.5)
            
            return surface
            
        finally:
            # Restore the original random state if we changed it
            if seed is not None:
                np.random.set_state(original_state)
    
    @staticmethod
    def generate_water_probability(grid_height, grid_width, num_sources=2, noise_scale=0.05):
        """
        Generate water probability map using distinct Gaussian distributions.
        Similar to the original _create_distribution method but with random centers.
        """
        prob_map = np.zeros((grid_height, grid_width))
        
        for _ in range(num_sources):
            # Random center for Gaussian
            mu = np.array([
                np.random.randint(0, grid_height),
                np.random.randint(0, grid_width)
            ])
            
            # Create covariance matrix with controlled spread
            spread = np.random.uniform(2, 4)  # Random spread size
            rotation = np.random.uniform(0, 2*np.pi)  # Random rotation
            
            # Create rotation matrix
            c, s = np.cos(rotation), np.sin(rotation)
            R = np.array([[c, -s], [s, c]])
            
            # Base covariance with random spread
            base_sigma = np.array([[spread, 0], [0, spread]])
            
            # Apply rotation to covariance
            sigma = R @ base_sigma @ R.T
            
            # Generate grid coordinates
            x = np.arange(0, grid_width)
            y = np.arange(0, grid_height)
            x, y = np.meshgrid(x, y)
            pos = np.dstack((y, x))
            
            # Calculate Gaussian values
            n = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
            inv_sigma = np.linalg.inv(sigma)
            diff = pos - mu
            exponent = -0.5 * np.einsum('...k,kl,...l->...', diff, inv_sigma, diff)
            gaussian = n * np.exp(exponent)
            
            prob_map += gaussian
        
        # Normalize and threshold to make distributions more distinct
        prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())
        prob_map[prob_map < 0.15] = 0  # Cut off low probabilities
        
        # clearn landing zone
        prob_map = Utils.clear_landing_zone(prob_map, grid_height, grid_width)
        
        return prob_map

    @staticmethod
    def generate_gold_probability(grid_height, grid_width, num_veins=3, noise_scale=0.05):
        """
        Generate gold probability map with guaranteed veins.
        """
        prob_map = np.zeros((grid_height, grid_width))
        min_vein_length = 7  # Minimum length of each vein
        
        veins_created = 0
        max_attempts = 50  # Prevent infinite loops
        attempts = 0
        
        while veins_created < num_veins and attempts < max_attempts:
            # Start point for vein
            start_y = np.random.randint(0, grid_height)
            start_x = np.random.randint(0, grid_width)
            
            # Choose a primary direction
            direction = np.random.choice(['vertical', 'diagonal1', 'diagonal2', 'horizontal'])
            
            # Direction probabilities
            if direction == 'vertical':
                dy_probs = [0.7, 0.2, 0.1]  # High probability of moving vertically
                dx_probs = [0.2, 0.6, 0.2]  # Some horizontal movement
            elif direction == 'horizontal':
                dy_probs = [0.2, 0.6, 0.2]
                dx_probs = [0.7, 0.2, 0.1]
            elif direction == 'diagonal1':
                dy_probs = [0.6, 0.2, 0.2]
                dx_probs = [0.6, 0.2, 0.2]
            else:  # diagonal2
                dy_probs = [0.2, 0.2, 0.6]
                dx_probs = [0.6, 0.2, 0.2]
            
            # Track positions for this vein
            current_y, current_x = start_y, start_x
            vein_positions = [(current_y, current_x)]
            vein_length = 0
            
            # Generate the vein
            for _ in range(12):  # Maximum steps
                # Move in the primary direction
                dy = np.random.choice([-1, 0, 1], p=dy_probs)
                dx = np.random.choice([-1, 0, 1], p=dx_probs)
                
                new_y = np.clip(current_y + dy, 0, grid_height - 1)
                new_x = np.clip(current_x + dx, 0, grid_width - 1)
                
                # Only add position if it moved
                if (new_y, new_x) != (current_y, current_x):
                    vein_positions.append((new_y, new_x))
                    current_y, current_x = new_y, new_x
                    vein_length += 1
            
            # Only add vein if it's long enough
            if vein_length >= min_vein_length:
                # Add the vein to the probability map
                for pos_y, pos_x in vein_positions:
                    # Create a small area of influence around each position
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            y = np.clip(pos_y + dy, 0, grid_height - 1)
                            x = np.clip(pos_x + dx, 0, grid_width - 1)
                            distance = np.sqrt(dy**2 + dx**2)
                            prob_map[y, x] = max(prob_map[y, x], 1.0 * np.exp(-2.0 * distance))
                
                veins_created += 1
            
            attempts += 1
        
        # Normalize
        if prob_map.max() > 0:  # Avoid division by zero
            prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())

        # clearn landing zone
        prob_map = Utils.clear_landing_zone(prob_map, grid_height, grid_width)
        
        return prob_map

    @staticmethod    
    def clear_landing_zone(probability_map, grid_height, grid_width, zone_size=3):
        """Zero out probabilities in the landing zone"""
        center_y = grid_height // 2
        center_x = grid_width // 2
        half_zone = zone_size // 2
        
        probability_map[
            center_y - half_zone : center_y + half_zone + 1,
            center_x - half_zone : center_x + half_zone + 1
        ] = 0
        
        return probability_map

    @staticmethod
    def generate_ground_truth(probability_map, noise_factor=0.2, threshold=0.2, existing_resources=None):
        """
        Generate ground truth resource map directly from probabilities.
        With noise_factor=0 and threshold=0, returns True for any non-zero probability.
        """
        if noise_factor == 0 and threshold == 0:
            return probability_map > 0
        else:
            # Add noise to probabilities
            noisy_probs = probability_map + np.random.normal(
                0, noise_factor, probability_map.shape
            )
            
            # Clip to valid probability range
            noisy_probs = np.clip(noisy_probs, 0, 1)
            
            # Generate ground truth based on noisy probabilities
            ground_truth = np.random.random(probability_map.shape) < noisy_probs
            
            # Apply threshold
            ground_truth &= (probability_map > threshold)

            # If there are existing resources, ensure no overlap
            if existing_resources is not None:
                ground_truth &= ~existing_resources
            
            return ground_truth
    
    @staticmethod
    def get_resource_probability(env, position, resource_type=None):
        """
        Get the probability of resources at a given position.
        Useful for the agent's observations.
        
        Args:
            position (tuple): (row, col) position to check
            resource_type (str): Optional, "water" or "gold". If None, returns both.
        
        Returns:
            float or tuple: Probability value(s) for specified resource(s)
        """
        i, j = position
        
        if resource_type == "water":
            return env.water_probability[i, j]
        elif resource_type == "gold":
            return env.gold_probability[i, j]
        else:
            return (env.water_probability[i, j], env.gold_probability[i, j])
    
    @staticmethod
    def _update_resource_probabilities(env, center_i, center_j):
        """
        Update water probabilities and confidence using discrete rings based on environment size.
        
        Args:
            env: The environment instance
            center_i, center_j: Center coordinates of the gathering action
        """
        # Calculate ring sizes based on environment dimensions
        env_size = env.grid_height  # Since it's a square environment
        first_ring_size = max(1, int(env_size * 0.1 / 2))   # 10% total width → ~5% from center
        second_ring_size = max(2, int(env_size * 0.2 / 2))  # 20% total width → ~10% from center
        third_ring_size = max(3, int(env_size * 0.3 / 2))   # 30% total width → ~15% from center
        
        # Update factors for each ring
        first_ring_factor = 0.1
        second_ring_factor = 0.05
        third_ring_factor = 0.01

        # Confidence factors for each ring
        first_ring_conf = 0.3
        second_ring_conf = 0.15
        third_ring_conf = 0.05
        
        # Get gathering result
        has_water = env.water_ground_truth[center_i, center_j]
        
        # Update center cell with certainty
        env.water_probability[center_i, center_j] = 1.0 if has_water else 0.0
        env.confidence_map[center_i, center_j] = 1.0  # Complete confidence in gathered cell
        
        # Calculate bounds for each ring
        for i in range(env.grid_height):
            for j in range(env.grid_width):
                if i == center_i and j == center_j:
                    continue  # Skip center cell as it's already updated
                    
                # Calculate Manhattan distance from center
                distance = max(abs(i - center_i), abs(j - center_j))
                
                # Determine which ring the cell belongs to and apply appropriate factors
                if distance <= first_ring_size:
                    prob_factor = first_ring_factor
                    conf_factor = first_ring_conf
                elif distance <= second_ring_size:
                    prob_factor = second_ring_factor
                    conf_factor = second_ring_conf
                elif distance <= third_ring_size:
                    prob_factor = third_ring_factor
                    conf_factor = third_ring_conf
                else:
                    continue  # Skip cells outside the third ring
                
                # Update probability based on water presence and factor
                if has_water:
                    env.water_probability[i, j] = min(1.0, env.water_probability[i, j] + prob_factor)
                else:
                    env.water_probability[i, j] = max(0.0, env.water_probability[i, j] - prob_factor)
                
                # Update confidence - take maximum between current and new confidence
                env.confidence_map[i, j] = max(env.confidence_map[i, j], conf_factor)

    @staticmethod
    def calculate_height(pos, height_map):
        """
        Returns the height at the given position from the provided height map
        Args:
            pos: tuple (x, y) representing the position
            height_map: numpy array containing height values
        Returns:
            float: height value at the given position
        """
        i, j = pos
        return height_map[i, j]

    @staticmethod
    def calculate_dust(pos, dust_map):
        """
        Returns the dust value at the given position from the dust map
        Args:
            pos: tuple (x, y) representing the position
            dust_map: numpy array containing dust values
        Returns:
            float: dust value at the given position
        """
        i, j = pos
        return dust_map[i, j]

    @staticmethod
    def calculate_sunlight_map(grid_height, grid_width, height_map, day_step, lunar_day=30):
        """Calculate a 2D sunlight map for the entire grid based on the lunar day-night cycle."""
        # Convert day_step to a fraction of the lunar cycle (0 to 1)
        time_fraction = (day_step % lunar_day) / lunar_day

        # Calculate base sunlight level using a sinusoidal cycle
        # Peaks at midday (time_fraction = 0.5), troughs at midnight (0 or 1)
        base_light = 0.5 * (1 + np.sin(2 * np.pi * (time_fraction - 0.25)))

        # Create the sunlight map
        sunlight = np.zeros((grid_height, grid_width))
        for i in range(grid_height):
            for j in range(grid_width):
                # Rescale height from [-50, 50] to [0, 1]
                normalized_height = (height_map[i, j] + 50) / 100
                # Calculate height factor based on normalized height
                height_factor = normalized_height + 0.5
                # Apply height adjustment to base light
                sunlight[i, j] = base_light * height_factor

        # Ensure values are clamped between [0, 1]
        return np.clip(sunlight, 0, 1)

    @staticmethod
    def calculate_sunlight_level(sunlight_map, i, j):
        """Get the sunlight level for a specific cell from the sunlight map."""
        return sunlight_map[i, j]

    @staticmethod
    def calculate_bat(current_pos, next_pos, current_bat_level, batt_capacity, sunlight_map, height_map, dust_map, action, num_solar_panels=3):
        """
        Calculate new battery level based on energy consumption and generation.
        
        Args:
            current_pos: Current position tuple (x, y)
            next_pos: Next position tuple (x, y)
            current_bat_level: Current battery level in Wh
            batt_capacity: Maximum battery capacity in Wh
            sunlight_map: 2D array of sunlight levels
            height_map: 2D array of height values
            dust_map: 2D array of dust levels
            action: Integer representing the action taken
            num_solar_panels: Number of solar panels (default 3)
        
        Returns:
            [next_bat_level, done]: New battery level and done flag
        """
        # 1. Calculate energy generation
        sunlight = Utils.calculate_sunlight_level(sunlight_map, next_pos[0], next_pos[1])
        daily_solar_output = 272.2 * 24 * num_solar_panels  # Wh per day per panel
        energy_generated = daily_solar_output * sunlight
        
        # 2. Calculate energy consumption
        energy_consumed = 1200  # Base consumption
        
        if action != 4:  # If not staying still
            # Movement energy calculation
            dust_level = Utils.calculate_dust(next_pos, dust_map)
            dust_factor = 1 + (dust_level * 0.5)  # 1-1.25x multiplier (since dust level is 0-0.5)
            
            height_diff = abs(Utils.calculate_height(next_pos, height_map) - 
                            Utils.calculate_height(current_pos, height_map))
            height_factor = 0.5 + (height_diff / 100)  # 0.5-1.5x multiplier (since height diff is -50 to 50)
            
            movement_energy = 13890 * dust_factor * height_factor
            energy_consumed += movement_energy
        
        if action == 5:  # Gathering action
            energy_consumed += 20000  # Additional energy for gathering
        
        # 3. Calculate new battery level
        next_bat_level = current_bat_level + energy_generated - energy_consumed
        
        # 4. Check battery limits
        if next_bat_level <= 0:
            return [0, True]  # Battery depleted
        elif next_bat_level > batt_capacity:
            next_bat_level = batt_capacity
        
        return [next_bat_level, False]
    
    @staticmethod
    def calculate_death_probability(current_day, max_probability=0.05):
        """
        Calculate probability of random death using sigmoid function.
        Probability increases from 0 to max_probability over 365 days.
        
        Args:
            current_day (int): Current day number
            max_probability (float): Maximum probability of death (default 0.05 or 5%)
        
        Returns:
            float: Probability of death for the current day
        """
        k = 0.02  # Steepness of sigmoid curve
        x0 = 182.5  # Midpoint (day 365/2)
        return max_probability / (1 + np.exp(-k * (current_day - x0)))

    @staticmethod
    def calculate_stuck_probability(dust_level):
        """
        Calculate probability of getting stuck based on dust level using sigmoid function.
        
        Args:
            dust_level (float): Current dust level (0 to 0.5m)
        
        Returns:
            float: Probability of getting stuck
        """
        k = 15  # Steepness of sigmoid curve
        x0 = 0.25  # Midpoint (0.5m/2)
        return 0.5 / (1 + np.exp(-k * (dust_level - x0)))  # Max probability of 0.5

    @staticmethod
    def check_crash(current_height, next_height):
        """
        Check if height difference would cause a crash.
        
        Args:
            current_height (float): Current position height
            next_height (float): Next position height
        
        Returns:
            bool: True if crash would occur
        """
        return abs(next_height - current_height) >= 25

    @staticmethod
    def epsilon_greedy(env, action_values, state, epsilon=0.1, is_q_values=False):
        """
        Returns an action using epsilon-greedy strategy, compatible with both policies and Q-values.
        If is_q_values is True, action_values is treated as Q-values; otherwise, as a policy.
        """
        if is_q_values:
            # Treat action_values as Q-values and construct an epsilon-greedy policy
            q_values = action_values  # Here, action_values is already the Q-values for this specific state
            action_probs = np.ones(env.action_space.n) * epsilon / env.action_space.n
            best_action = np.argmax(q_values)
            action_probs[best_action] += 1.0 - epsilon
        else:
            # Treat action_values as a policy and apply epsilon-greedy using state_idx
            state_idx = Utils._state_to_index(env, state)
            policy = action_values[state_idx]
            action_probs = (1 - epsilon) * policy + (epsilon / env.action_space.n)

        # Choose an action based on the adjusted probabilities
        return np.random.choice(np.arange(env.action_space.n), p=action_probs)

    @staticmethod
    def draw_arrow(screen, x, y, direction, size=50, color=(0, 0, 0)):
        """Draws an arrow at the given (x, y) location, pointing in the given direction."""
        half_size = size // 2
        arrow_head_size = size * 0.6  # Adjust size of the arrowhead
        shaft_length = size * 0.35  # Length of the shaft

        # Calculate the start and end points of the shaft
        if direction == 0:  # Left
            start = (x + half_size, y)
            end = (x - shaft_length, y)
            arrow_tip = (x - half_size, y)
            arrow_head = [
                (arrow_tip[0], arrow_tip[1] - arrow_head_size // 2),
                (arrow_tip[0], arrow_tip[1] + arrow_head_size // 2),
                (arrow_tip[0] - arrow_head_size, arrow_tip[1]),
            ]

            pygame.draw.line(
                screen, color, start, end, 5
            )  # Draw the shaft of the arrow (line)
            pygame.draw.polygon(
                screen, color, arrow_head
            )  # Draw the arrowhead (triangle)

        elif direction == 1:  # Right
            start = (x - half_size, y)
            end = (x + shaft_length, y)
            arrow_tip = (x + half_size, y)
            arrow_head = [
                (arrow_tip[0], arrow_tip[1] - arrow_head_size // 2),
                (arrow_tip[0], arrow_tip[1] + arrow_head_size // 2),
                (arrow_tip[0] + arrow_head_size, arrow_tip[1]),
            ]

            pygame.draw.line(
                screen, color, start, end, 5
            )  # Draw the shaft of the arrow (line)
            pygame.draw.polygon(
                screen, color, arrow_head
            )  # Draw the arrowhead (triangle)

        elif direction == 2:  # Up
            start = (x, y + half_size)
            end = (x, y - shaft_length)
            arrow_tip = (x, y - half_size)
            arrow_head = [
                (arrow_tip[0] - arrow_head_size // 2, arrow_tip[1]),
                (arrow_tip[0] + arrow_head_size // 2, arrow_tip[1]),
                (arrow_tip[0], arrow_tip[1] - arrow_head_size),
            ]

            pygame.draw.line(
                screen, color, start, end, 5
            )  # Draw the shaft of the arrow (line)
            pygame.draw.polygon(
                screen, color, arrow_head
            )  # Draw the arrowhead (triangle)

        elif direction == 3:  # Down
            start = (x, y - half_size)
            end = (x, y + shaft_length)
            arrow_tip = (x, y + half_size)
            arrow_head = [
                (arrow_tip[0] - arrow_head_size // 2, arrow_tip[1]),
                (arrow_tip[0] + arrow_head_size // 2, arrow_tip[1]),
                (arrow_tip[0], arrow_tip[1] + arrow_head_size),
            ]

            pygame.draw.line(
                screen, color, start, end, 5
            )  # Draw the shaft of the arrow (line)
            pygame.draw.polygon(
                screen, color, arrow_head
            )  # Draw the arrowhead (triangle)

        elif direction == 4 or direction == 5:  # Stay in place/gather
            # Draw a circle
            pygame.draw.circle(screen, color, (x, y), size // 3, 5)

    @staticmethod
    def render_optimal_policy(
        env, policy, save_image=False, image_filename="policy_visualization.png"
    ):
        """Renders the optimal policy using arrows for GeosearchEnv."""
        if env.render_mode != 'human' or env.screen is None:
            print("Setting render_mode to 'human' and initializing rendering.")
            env.render_mode = 'human'
            env._init_render()

        if env.screen is None:
            print(
                "Warning: Cannot render policy without display mode. Set render_mode='human' when creating environment."
            )
            return

        Utils._ensure_directory_exists(image_filename)
        arrow_mapping = {
            0: 2,
            1: 3,
            2: 0,
            3: 1,
            4: 4,  # Stay in place -> Circle
            5: 5,
        }  # Mapping for directions: Left, Right, Up, Down

        # set the days to show for the policy visualization
        days_to_show = [0, 7, 15, 22]
        original_day = env.current_day

        for day in days_to_show:
            env.current_day = day
            env.screen.fill(env.colors["white"])  # Clear the screen

            # Draw grid cells with lunar characteristics
            for i in range(env.grid_height):
                for j in range(env.grid_width):
                    # Use the environment's color calculation method    
                    cell_color = env._get_cell_color(i, j)

                    # Draw the cell
                    pygame.draw.rect(
                        env.screen,
                        cell_color,
                        pygame.Rect(
                            j * env.cell_size,
                            i * env.cell_size,
                            env.cell_size,
                            env.cell_size,
                        ),
                    )
                    pygame.draw.rect(
                        env.screen,
                        env.colors["black"],
                        pygame.Rect(
                            j * env.cell_size,
                            i * env.cell_size,
                            env.cell_size,
                            env.cell_size,
                        ),
                        1,
                    )

                    # Draw arrows based on the optimal policy
                    state_idx = i * env.grid_width + j
                    if state_idx < len(policy):
                        optimal_action = np.argmax(policy[state_idx])
                        direction = arrow_mapping[optimal_action]
                        Utils.draw_arrow(
                            env.screen,
                            j * env.cell_size + env.cell_size // 2,
                            i * env.cell_size + env.cell_size // 2,
                            direction,
                            size=env.cell_size * 0.4,
                        )

            # Add title showing lunar day
            font = pygame.font.Font(None, 36)
            title_text = font.render(f"Lunar Day {day}", True, env.colors["white"])
            env.screen.blit(title_text, (10, 10))

            pygame.display.flip()

            if save_image:
                Utils.save_image(env.screen, filename=f"{image_filename}_day{day}.png")
            
            # Wait for click or key press
            print("Press any key while in pygame window to continue...")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                        waiting = False
                    elif event.type == pygame.QUIT:
                        env.close()
                        return

        # Restore original hour
        env.current_day = original_day

    @staticmethod
    def create_gif(frames, filename="gameplay.gif", duration=100):
        """Creates a GIF from a list of Pygame frames."""
        Utils._ensure_directory_exists(filename)
        pil_images = [Image.fromarray(frame) for frame in frames]
        pil_images[0].save(
            filename,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0,
        )
        print(f"Gameplay GIF saved as {filename}")

    @staticmethod
    def save_image(screen, filename="policy_visualization.png"):
        """Saves the current Pygame screen as an image."""
        Utils._ensure_directory_exists(filename)
        frame = pygame.surfarray.array3d(screen)
        image = Image.fromarray(
            np.transpose(frame, (1, 0, 2))
        )  # Convert for Pillow compatibility
        image.save(filename)
        print(f"Optimal policy visualization saved as {filename}")

    @staticmethod
    def run_optimal_policy(
        env,
        agent,  # Pass the DQN agent instead of policy
        episodes=5,
        max_steps=50,
        delay_ms=66,
        save_gif=False,
        gif_filename="gameplay.gif",
    ):
        """Simulate episodes following the trained DQN policy."""
        # Ensure rendering is initialized
        if env.render_mode is not None and env.screen is None:
            env._init_render()

        if env.screen is None:
            print("Warning: Cannot render policy without display mode.")
            return

        Utils._ensure_directory_exists(gif_filename)
        frames = []
        total_reward = 0  # Track total reward across all episodes

        for episode in range(episodes):
            state, info = env.reset()
            terminated = False
            episode_reward = 0
            steps = 0

            while not terminated:
                # Render and capture frame if saving GIF
                env.render()
                if save_gif:
                    frame = pygame.surfarray.array3d(env.screen)
                    frames.append(np.transpose(frame, (1, 0, 2)))

                pygame.time.wait(delay_ms)

                # Use the DQN to select best action (no exploration)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(agent.process_state(state)).unsqueeze(0).to(agent.device)
                    action = agent.policy_net(state_tensor).max(1)[1].view(1, 1).item()

                # Take action in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Update state and accumulate reward
                state = next_state
                episode_reward += reward

                steps += 1
                if steps >= max_steps:
                    terminated = True

            total_reward += episode_reward

        # Save GIF if required
        if save_gif:
            Utils.create_gif(frames, filename=gif_filename, duration=delay_ms)

        avg_reward = total_reward / episodes
        print(f"Average reward following DQN policy: {avg_reward:.2f}")

    @staticmethod
    def plot_convergence(convergence_data, file_path="convergence_plot.png"):
        """Plots the convergence of mean reward over episodes."""
        Utils._ensure_directory_exists(file_path)
        plt.figure()
        plt.plot(convergence_data)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.title("Convergence of the Mean Reward over Episodes")
        plt.grid(True)
        plt.savefig(file_path)
        print(f"Convergence plot saved as {file_path}")

