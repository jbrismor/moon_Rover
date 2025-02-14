import os
from PIL import Image # gif creation
import numpy as np
from scipy.ndimage import gaussian_filter 

class Utils:
    @staticmethod
    def generate_height_map(grid_height=20, grid_width=20, smoothing=2.0, seed=None):
        """Generate a normalized height map with values between 0-1"""
        original_state = np.random.get_state()
        try:
            if seed is not None:
                np.random.seed(seed)
                
            # start with a flat terrain around 0.5
            height_map = np.full((grid_height, grid_width), 0.5)

            # add some small noise
            height_map += np.random.normal(0, 0.1, (grid_height, grid_width))
            
            # add mountains (much higher elevation changes)
            for _ in range(4):  # 2 mountain ranges
                height_map = Utils._add_terrain_feature(
                    height_map, 
                    elevation_range=(0.7, 0.9),  # higher elevation for mountains
                    feature_width=3,  # wider features
                    spread=True  # spread elevation to neighboring cells
                )
                
            # add two deep valley
            for _ in range(3):
                height_map = Utils._add_terrain_feature(
                    height_map, 
                    elevation_range=(-0.8, -0.6),  # deeper valley
                    feature_width=4,  # wider valley
                    spread=True
                )
            
            # light smoothing to blend edges
            height_map = gaussian_filter(height_map, sigma=0.5)
            
            # normalize but enhance contrast
            height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
            height_map = np.power(height_map, 0.7)  # Enhance contrast
            
            return height_map
        finally:
            np.random.set_state(original_state)

    @staticmethod
    def _add_terrain_feature(height_map, elevation_range, feature_width=2, spread=True):
        """Add a terrain feature (mountain or valley) to the height map"""
        grid_height, grid_width = height_map.shape
        
        # starting location and length
        start_x = np.random.randint(0, grid_width)
        start_y = np.random.randint(0, grid_height)
        length = np.random.randint(6, 12)  # longer features
        
        # direction
        direction = np.random.rand() * 2 * np.pi
        
        # add elevation changes
        x, y = start_x, start_y
        for _ in range(length):
            x_idx = int(np.clip(x, 0, grid_width-1))
            y_idx = int(np.clip(y, 0, grid_height-1))
            
            # add elevation change to main point
            elevation = np.random.uniform(*elevation_range)
            
            # apply elevation to an area around the point
            for dy in range(-feature_width, feature_width + 1):
                for dx in range(-feature_width, feature_width + 1):
                    ny = y_idx + dy
                    nx = x_idx + dx
                    
                    if 0 <= ny < grid_height and 0 <= nx < grid_width:
                        if spread:
                            # distance-based elevation falloff
                            distance = np.sqrt(dx**2 + dy**2)
                            falloff = 1 / (1 + distance)
                            height_map[ny, nx] += elevation * falloff
                        else:
                            height_map[ny, nx] += elevation
            
            # move in general direction with randomness
            x += np.cos(direction) + np.random.uniform(-0.3, 0.3)
            y += np.sin(direction) + np.random.uniform(-0.3, 0.3)
        
        return height_map

    @staticmethod
    def generate_dust_map(height_map, smoothing=2.0, dust_height_correlation=0.5):
        """
        Generate dust map with values between 0-0.5, 
        correlated with height (more dust in lower areas)
        """
        grid_height, grid_width = height_map.shape
        
        # create base random dust distribution
        base_dust = np.random.normal(0, 0.1, (grid_height, grid_width))
        dust_map = gaussian_filter(base_dust, sigma=smoothing)
        
        # normalize base dust to 0-1
        dust_map = (dust_map - dust_map.min()) / (dust_map.max() - dust_map.min())
        
        # create height influence (lower areas have more dust)
        height_influence = 1 - height_map
        
        # combine dust and height influence
        dust_map = (dust_height_correlation * height_influence + 
                   (1 - dust_height_correlation) * dust_map)
        
        # ensure range is within 0-1 and then scale to desired range
        dust_map = np.clip(dust_map, 0, 1)
        dust_map *= 0.5
        
        return dust_map
    
    @staticmethod
    def calculate_sunlight(current_day, height):
        """Calculate sunlight level based on day and height"""
        # convert current_day to a fraction of the lunar cycle
        time_fraction = (current_day % 30) / 30.0

        # calculate base sunlight level using a sinusodial cycle
        # peaks at full moon (15), troughs at new moon (0)
        base_sunlight = round(0.5 * (1 + np.sin(2 * np.pi * (time_fraction - 0.25))), 4)

        # height factor (higher = more sunlight)
        height_factor = 0.5 + height # range 0.5-1.5

        # apply height adjustment to base sunlight
        sunlight = base_sunlight * height_factor

        # ensurance values clamped between 0 and 1
        sunlight = np.clip(sunlight, 0, 1)

        return sunlight

    @staticmethod
    def calculate_energy_input(sunlight_level, num_solar_panels, daily_panel_input):
        """Calculate energy generated from solar panels"""
        daily_energy_input = sunlight_level * num_solar_panels * daily_panel_input
        return daily_energy_input

    @staticmethod
    def calculate_energy_consumption(dust, prev_height, potential_next_height, movement_base_energy, base_consumption):
        """Calculate energy cost of movement"""
        # dust factor
        dust_factor = 1 + 0.5 * dust # 1-1.25x multiplier

        # height difference and factor
        height_diff = abs(potential_next_height - prev_height)
        height_factor = 0.5 + height_diff # 0.5-1.5x multiplier

        # movement energy
        movement_energy = dust_factor * height_factor * movement_base_energy

        #base energy consumption
        energy_consumption = base_consumption + movement_energy
        return energy_consumption

    @staticmethod
    def generate_resource_probs(grid_height=20, grid_width=20, num_sources=3, min_spread=2, max_spread=4):
        """
        Generate resource probability map using Gaussian distributions.
        
        Args:
            grid_height (int): Height of the grid
            grid_width (int): Width of the grid
            num_sources (int): Number of resource clusters
            min_spread (float): Minimum spread of Gaussian distributions
            max_spread (float): Maximum spread of Gaussian distributions
        """
        prob_map = np.zeros((grid_height, grid_width))
        
        for _ in range(num_sources):
            # random center for gaussian
            mu = np.array([
                np.random.randint(0, grid_height),
                np.random.randint(0, grid_width)
            ])
            
            # create covariance matrix with controlled spread
            spread = np.random.uniform(min_spread, max_spread) # spread size
            rotation = np.random.uniform(0, 2*np.pi) # rotation angle
            
            # create rotation matrix
            c, s = np.cos(rotation), np.sin(rotation)
            R = np.array([[c, -s], [s, c]])
            
            # base covariance with random spread
            base_sigma = np.array([[spread, 0], [0, spread]])
            
            # apply rotation to covariance
            sigma = R @ base_sigma @ R.T
            
            # generate grid coordinates
            x = np.arange(0, grid_width)
            y = np.arange(0, grid_height)
            x, y = np.meshgrid(x, y)
            pos = np.dstack((y, x))
            
            # calculate Gaussian values
            n = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
            inv_sigma = np.linalg.inv(sigma)
            diff = pos - mu
            exponent = -0.5 * np.einsum('...k,kl,...l->...', diff, inv_sigma, diff)
            gaussian = n * np.exp(exponent)
            
            prob_map += gaussian
        
        # normalize and threshold
        prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())
        prob_map[prob_map < 0.15] = 0  # Cut off low probabilities
        
        # clear landing zone (center area)
        landing_radius = 2
        center_y, center_x = grid_height // 2, grid_width // 2
        y_indices, x_indices = np.ogrid[:grid_height, :grid_width]
        dist_from_center = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
        prob_map[dist_from_center < landing_radius] = 0
        
        return prob_map

    @staticmethod
    def generate_ground_truth(probability_map, noise_factor=0.2, threshold=0.3):
        """
        Generate ground truth resource map from probability distribution.
        
        Args:
            probability_map (np.ndarray): Base probability distribution
            noise_factor (float): Amount of random noise to add (0-1)
            threshold (float): Minimum probability for resource generation
            
        Returns:
            np.ndarray: Binary map of resource locations
        """
        # add controlled noise to probabilities
        noise = np.random.normal(0, noise_factor, probability_map.shape)
        noisy_probs = np.clip(probability_map + noise, 0, 1)
        
        # generate random values for comparison
        random_values = np.random.random(probability_map.shape)

        # adaptive threshold that's lower for higher probability areas
        scaled_threshold = threshold * (1 - probability_map)  
        
        # generate ground truth based on adaptive threshold
        ground_truth = (random_values < noisy_probs) & (noisy_probs > scaled_threshold)
        
        return ground_truth

    @staticmethod
    def update_resource_probs(env, center_i, center_j, influence_radius=3, update_strength=0.2):
        """
        Update resource probabilities based on gathering results.
        Assumes resources are clustered, so finding/not finding a resource 
        influences nearby probabilities.
        
        Args:
            env: Environment instance containing probability and ground truth maps
            center_i, center_j: Center coordinates of gathering action
            influence_radius: Radius of probability update effect
            update_strength: Base strength of probability updates (0-1)
        """
        # get gathering result
        has_resource = env.resource_ground_truth[center_i, center_j]
        
        # calculate update region bounds
        i_min = max(0, center_i - influence_radius)
        i_max = min(env.grid_height, center_i + influence_radius + 1)
        j_min = max(0, center_j - influence_radius)
        j_max = min(env.grid_width, center_j + influence_radius + 1)
        
        # create distance matrix for the update region
        y_indices, x_indices = np.ogrid[i_min:i_max, j_min:j_max]
        distances = np.sqrt((y_indices - center_i)**2 + (x_indices - center_j)**2)
        
        # calculate update factors: stronger at center, falling off with distance
        # using squared distance for more pronounced local effect
        update_factors = update_strength * (1 - (distances / influence_radius)**2)
        update_factors = np.clip(update_factors, 0, update_strength)
        
        # update region probabilities
        region = env.resource_prob_map[i_min:i_max, j_min:j_max]
        
        if has_resource:
            # found resource: increase nearby probabilities
            region += update_factors * (1 - region)  # ensure we don't exceed 1.0
            
            # set gathered location to certainty
            env.resource_prob_map[center_i, center_j] = 1.0
        else:
            # no resource: decrease probabilities
            region -= update_factors * region  # ensure we don't go below 0.0
            
            # set gathered location to certainty
            env.resource_prob_map[center_i, center_j] = 0.0
        
        # epdate the region in the main probability map
        env.resource_prob_map[i_min:i_max, j_min:j_max] = region

    @staticmethod
    def check_random_death(current_day, max_prob = 0.02):
        """Calculate probability of random death/failure using sigmoid function,
         with the probability increasing from 0 to max_prob over 360 days, returning
          (terminated, truncated)"""
        # sigmoid function parameters
        k = 0.02 # steepness of sigmoid curve
        x0 = 180 # midpoint

        # calculate current probability of death
        death_prob = max_prob / (1 + np.exp(-k * (current_day - x0)))

        # check if random death occurs
        if np.random.random() < death_prob:
            terminated = True
            truncated = False
            return terminated, truncated
        else:
            terminated = False
            truncated = False
            return terminated, truncated
        
    @staticmethod
    def check_stuck(dust_level):
        """Check if robot is stuck in dust"""
        # sigmoid function parameters
        k = 15 # steepness of sigmoid curve
        x0 = 0.15 # midpoint

        # calculate probability of getting stuck
        stuck_prob = 0.5 / (1 + np.exp(-k * (dust_level - x0)))

        # check if robot gets stuck
        if np.random.random() < stuck_prob:
            return True
        else:
            return False
        
    @staticmethod
    def check_crash(current_height, next_height):
        """Check if height difference would cause crash, returning True if crash"""
        return abs(next_height - current_height) >= 0.3

    @staticmethod
    def _ensure_directory_exists(file_path):
        """Ensures the parent directory of the given file path exists."""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

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