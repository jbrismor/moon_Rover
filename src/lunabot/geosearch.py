import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import importlib.resources
from .utils import Utils
from matplotlib import cm

class GeosearchEnv(gym.Env):
    def __init__(self, render_mode=None):
        """Initialize the Lunar Geosearch Environment"""
        super(GeosearchEnv, self).__init__()

        # grid setup
        self.grid_height = 20
        self.grid_width = 20

        # define action space
        self.action_space = spaces.Discrete(6)

        # track previous action for display
        self.prev_action = None

        # define observation space
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=np.array([0, 0]), 
                    high=np.array([self.grid_height-1, self.grid_width-1]), 
                    shape=(2,), dtype=np.float32),
            'battery': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'sunlight': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'dust': spaces.Box(low=0, high=0.5, shape=(1,), dtype=np.float32),
            'local_heights': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'local_probs': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'cardinal_probs': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            'location_depleted': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        # initialize observation
        self.obs = None

        # timeline configuration
        self.max_steps = 360 # lunar year
        self.month_length = 30 # approx of lunar month 
        self.current_day = 0
        self.last_month = 0 # to track month survival

        # ROBOT CONFIGURATION (initially None)
        self.agent_pos = None

        # initialize reward
        self.step_reward = 0

        # LUNAR CHARACTERISTICS
        self.height_map = Utils.generate_height_map(self.grid_height, self.grid_width)
        self.dust_map = Utils.generate_dust_map(self.height_map)

        # RESOURCE(S)
        self.resource_prob_map = Utils.generate_resource_probs()
        self.resource_ground_truth = Utils.generate_ground_truth(self.resource_prob_map)

        # resource tracking & rewards
        self.gathered_counts = {} # dictionary to track number of times each location is gathered
        self.resources_gathered = {'resource_1': {'count': 0, 'locations': set()}} # dictionary to track all resources gathered
        # self.max_gather_times = 4 # num times before rewards reach zero
        self.resource_1_reward = 5
        # self.gather_decay = 1.25
        self.repeat_gather_penalty = -0.1

        # initialize observation variables (no starting location until reset)
        self.cardinal_probs = None
        self.local_heights = None
        self.local_probs = None
        self.dust = None
        self.sunlight = None

        # ENERGY CONSTANTS (simplified)
        self.battery_capacity = 1 # percentage
        self.num_solar_panels = 3
        self.daily_panel_input = 0.05 # per panel per day (impacted by sunlight level)
        self.base_consumption = 0.01 # operating costs
        self.movement_base_energy = 0.15 # (impacted by height and dust factors)
        self.gathering_energy = 0.2
        self.battery_level = self.battery_capacity

        # stuck tracking
        self.is_stuck = False
        self.stuck_days = 0

        # rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.robot_image = None

        # colors
        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "resource": (15, 94, 156),
            "base_gray": (128, 128, 128),
        }

    def step(self, action):
        """Apply action and advance time by one day"""
        # # Initialize debug variables
        # self.prev_reward = self.total_reward
        # reward_after_base_penalty = None
        # reward_after_month_reward = None
        # reward_after_high_battery_penalty = None
        # reward_after_low_battery_penalty = None
        # self.reward_after_stuck_penalty = None
        # self.reward_after_height_penalty = None
        # self.reward_after_dust_penalty = None
        # self.reward_after_1_direction_reward = None
        # self.reward_after_2_direction_reward = None
        # self.reward_before_gathering = None
        # self.reward_after_gathering = None
         
        # advance time
        self.current_day += 1

        # base reward for each step
        self.step_reward = -0.01

        # #DEBUG
        # reward_after_base_penalty = self.total_reward

        # store action for rendering
        self.prev_action = action

        # check for month survival
        current_month = self.current_day // self.month_length
        if current_month > self.last_month:
            self.step_reward += 0.5 # reward for surviving a month
            # reward_after_month_reward = self.total_reward # DEBUG
            self.last_month = current_month
            # reward_after_month_reward = "N/A" # DEBUG

        # check for random death/failure
        terminated, truncated = Utils.check_random_death(self.current_day)

        if terminated:
            self._update_observation()
            return self.obs_dict, self.step_reward, terminated, truncated, {}

        # set prev_height and battery level
        self.prev_height = self.current_height
        self.prev_battery_level = self.battery_level

        # handle different actions
        if action in [0, 1, 2, 3]:  # movement actions
            terminated, truncated = self._handle_movement(action)
        elif action == 4:  # stay in place
            self._handle_stay()
        elif action == 5:  # gathering
            self._handle_gather()

        # apply battery level penalties
        if self.battery_level > 0.95:
            self.step_reward -= 0.01  # penalty for keeping battery too full
            # reward_after_high_battery_penalty = self.total_reward # DEBUG
        elif self.battery_level < 0.1:
            self.step_reward -= 0.05  # penalty for letting battery get too low
            # reward_after_low_battery_penalty = self.total_reward # DEBUG
        
        # check for episode end due to max steps
        if self.current_day >= self.max_steps:
            terminated = True
            truncated = True
        
        # debugging info
        info = {
            'current_day': self.current_day,
            'resources': self.resources_gathered,
            'battery_level': self.battery_level,
            'stuck': self.is_stuck
        }
        
        # update observation
        self._update_observation()

        # # DEBUG
        # if terminated:
        #     print(f"Episode terminated! Reason: {'Stuck' if self.stuck_days >= 10 else 'Crash' if Utils.check_crash(self.prev_height, self.current_height) else 'Unknown'}")
    
        # # DEBUG
        # if (self.total_reward - self.prev_reward) > 5 or (self.total_reward - self.prev_reward) < -0.6:  # More than worst-case per-step
        #     print(f"Large reward change detected:")
        #     print(f"Previous reward: {self.prev_reward}")
        #     print(f"Action: {action}")
        #     print(f"Current reward: {self.total_reward}")
        #     print(f"Change: {self.total_reward - self.prev_reward}")
        #     print(f"Current state:")
        #     print(f"- Day: {self.current_day}")
        #     print(f"- Battery: {self.battery_level}")
        #     print(f"- Dust: {self.dust}")
        #     print(f"- Height: {self.current_height}")
        #     print(f"- Stuck: {self.is_stuck}")
        #     if self.is_stuck:
        #         print(f"- Stuck days: {self.stuck_days}")
        
        return self.obs_dict, self.step_reward, terminated, truncated, info    

    def reset(self, seed=None, options=None):
        """Reset the environment with random starting position"""
        if seed is not None:
            np.random.seed(seed)

        # reset agent position
        self.agent_pos = self._generate_starting_position()

        # reset timeline tracking
        self.current_day = 0
        self.last_month = 0

        # reset rewards
        self.step_reward = 0

        # generate new dust distribution
        self.dust_map = Utils.generate_dust_map(self.height_map)

        # generate new resource distribution
        self.resource_prob_map = Utils.generate_resource_probs()
        self.resource_ground_truth = Utils.generate_ground_truth(self.resource_prob_map)

        # reset resource tracking
        self.gathered_counts = {}
        self.resources_gathered = {'resource_1': {'count': 0, 'locations': set()}}

        # reset cardinal probs
        self.cardinal_probs = self._cardinal_sums()

        # reset battery level
        self.battery_level = self.battery_capacity

        # reset stuck tracking
        self.is_stuck = False
        self.stuck_days = 0

        # OBSERVATION
        # get current height
        self.current_height = self.height_map[self.agent_pos]

        # calculate sunlight level
        self.sunlight = Utils.calculate_sunlight(self.current_day, self.current_height)

        # get current dust level
        self.dust = self.dust_map[self.agent_pos]

        # get local heights
        self.local_heights = self._get_local_values(self.height_map, self.agent_pos)

        # get local probs
        self.local_probs = self._get_local_values(self.resource_prob_map, self.agent_pos)

        # calculate and update cardinal probs
        self.cardinal_probs = self._cardinal_sums()

        # # DEBUG
        # self.prev_reward = self.total_reward

        # update observation dictionary
        self._update_observation()

        return self.obs_dict, {}

    def render(self):
        """Render the environment"""
        # if render mode is None, do nothing
        if self.render_mode is None:
            return
        
        # initialize pygame components if not already initialized
        self._init_render()

        # fill screen with white
        self.screen.fill(self.colors["white"])

        # set font size for within cell text
        font = pygame.font.Font(None, 20)

        # draw grid cells with characteristics
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # get height of cell
                height_value = self.height_map[i, j]

                # get color of cell
                cell_color = self._get_cell_color(i, j, height_value)

                # draw cell
                pygame.draw.rect(self.screen, cell_color, pygame.Rect(
                        j * self.cell_size,
                        i * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

                # draw grid lines
                pygame.draw.rect(self.screen, self.colors["black"], pygame.Rect(
                        j * self.cell_size,
                        i * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                    1,
                )

                # text color based on height and cell color
                text_color = self._get_height_text_color(height_value, cell_color)

                # draw height number with colored text
                height_text = f"{height_value:.2f}"
                text_surface = font.render(height_text, True, text_color)
                text_rect = text_surface.get_rect(
                    center=(
                        j * self.cell_size + self.cell_size // 2,
                        i * self.cell_size + self.cell_size // 2,
                    )
                )
                self.screen.blit(text_surface, text_rect)

        # draw robot with safe size calculation (robot is larger on higher terrain)
        height_at_pos = self.height_map[self.agent_pos]
        base_size = 25
        size_factor = 0.4 + height_at_pos
        adjusted_size = max(10, int(base_size * size_factor)) # size of robot ranges from 10 to 35 based on height

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

        # set font size for display info
        font = pygame.font.Font(None, 36)

        # resource info
        resource_count = self.resources_gathered['resource_1']['count']

        # action info
        action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Stay", 5: "Gather"}

        # display information
        day_text = font.render(f"Lunar Day: {self.current_day}", True, self.colors["white"])
        height_text = font.render(f"Height: {self.height_map[self.agent_pos]:.1f}m", True, self.colors["white"])
        dust_text = font.render(f"Dust: {self.dust_map[self.agent_pos]:.2f}m", True, self.colors["white"])
        battery_text = font.render(f"Battery: {self.battery_level:.2f}%", True, self.colors["white"])
        stuck_text = font.render(f"Stuck: {str(self.is_stuck)}", True, self.colors["white"])
        action_text = font.render(f"Action: {action_names.get(self.prev_action, 'None')}", True, self.colors["white"])
        resource_text = font.render(f"Resource: {resource_count}", True, self.colors["white"])

        # Display all text on the screen
        self.screen.blit(day_text, (10, 10))
        self.screen.blit(height_text, (10, 50))
        self.screen.blit(dust_text, (10, 90))
        self.screen.blit(battery_text, (10, 130))
        self.screen.blit(stuck_text, (10, 170))
        self.screen.blit(action_text, (10, 210))
        self.screen.blit(resource_text, (10, 250))

        # update display
        pygame.display.flip()

        # maintain target FPS
        self.clock.tick(10)

    def close(self):
        """Close the environment"""
        if self.screen is None:
            return
        pygame.quit()
        self.screen = None
        self.clock = None
        self.robot_image = None

    def _init_render(self):
        """Initialie Pygame components if not already initialized"""
        if self.screen is None and self.render_mode is not None:
            pygame.init()

            # screen dimensions
            self.screen_width = 700
            self.screen_height = 700
            self.cell_size = 35 # size of each cell in pixels

            # create pygame window and set title
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Lunar Geosearch Environment")
            self.clock = pygame.time.Clock()

            # using importlib.resources to load the robot image
            with importlib.resources.path("lunabot", "robot.png") as robot_path:
                self.robot_image = pygame.image.load(str(robot_path))
                self.robot_image = pygame.transform.scale(self.robot_image, (20, 20))

    def _generate_starting_position(self):
        """
        Generate a starting position for the agent in a 3x3 square around 
        the center of the grid.
        """
        # find the middle of the grid
        mid_height = self.grid_height // 2
        mid_width = self.grid_width // 2
        
        # calculate bounds for starting area
        min_row = max(0, mid_height - 1)
        max_row = min(self.grid_height - 1, mid_height + 1)
        min_col = max(0, mid_width - 1)
        max_col = min(self.grid_width - 1, mid_width + 1)
        
        # generate random position within bounds
        start_pos = (
            np.random.randint(min_row, max_row + 1),
            np.random.randint(min_col, max_col + 1)
        )
        
        return start_pos

    def _cardinal_sums(self):
        """
        Calculate the sum of resource probabilities in each cardinal direction
        from the agent's current position.
        
        Returns:
        numpy array of shape (4,) containing raw sums in order: [north, south, west, east]
        """
        i, j = self.agent_pos
        rows, cols = self.resource_prob_map.shape
        
        # initialize sums for each direction
        cardinal_sums = np.zeros(4)
        
        # north (all cells above current position)
        if i > 0:
            cardinal_sums[0] = np.sum(self.resource_prob_map[0:i, j])
        
        # south (all cells below current position)
        if i < rows - 1:
            cardinal_sums[1] = np.sum(self.resource_prob_map[i+1:, j])
        
        # west (all cells to the left)
        if j > 0:
            cardinal_sums[2] = np.sum(self.resource_prob_map[i, 0:j])
        
        # east (all cells to the right)
        if j < cols - 1:
            cardinal_sums[3] = np.sum(self.resource_prob_map[i, j+1:])

        # Normalize the sums to be between 0 and 1
        if np.sum(cardinal_sums) > 0:
            cardinal_sums = cardinal_sums / np.max(cardinal_sums)
        
        return cardinal_sums

    def _get_local_values(self, value_map, agent_pos, default=0.5):
        """
        Returns a list of local values around the agent in the order:
        [center, above, below, left, right].
        
        Parameters:
        - value_map: 2D numpy array of height values.
        - agent_pos: Tuple (i, j) indicating the agent's position.
        - default: The value to use for out-of-bound neighbors.
        """
        i, j = agent_pos
        rows, cols = value_map.shape

        # center value (the cell the agent is in)
        center = value_map[i, j]
        
        # above: only valid if agent is not on the first row
        above = value_map[i - 1, j] if i - 1 >= 0 else default

        # below: only valid if agent is not on the last row
        below = value_map[i + 1, j] if i + 1 < rows else default

        # left: only valid if agent is not in the first column
        left = value_map[i, j - 1] if j - 1 >= 0 else default

        # right: only valid if agent is not in the last column
        right = value_map[i, j + 1] if j + 1 < cols else default

        return [center, above, below, left, right]
    
    def _handle_movement(self, action):
        """Handle movement actions (Up, Down, Left, Right)"""
        # current position
        i, j = self.agent_pos
        
        # movement directions mapping
        directions = {
            0: (-1, 0, 'north'),  # Up
            1: (1, 0, 'south'),   # Down
            2: (0, -1, 'west'),   # Left
            3: (0, 1, 'east')     # Right
        }
        
        # get direction deltas and name
        di, dj, direction = directions[action]
        
        # check boundary conditions
        if not self._is_valid_move(i + di, j + dj):
            return False, False
        
        # potential next position
        potential_next_pos = (i + di, j + dj)
        
        # handle energy and movement
        terminated, truncated = self._process_movement(potential_next_pos, action)
        
        return terminated, truncated

    def _is_valid_move(self, i, j):
        """Check if the move is within grid boundaries"""
        valid = (0 <= i < self.grid_height) and (0 <= j < self.grid_width)
        if not valid:
            self.step_reward -= 0.01  # small extra penalty for invalid moves
        return valid

    def _process_movement(self, potential_next_pos, action):
        """Process the movement including energy, stuck checks, and updates"""
        # potential next height
        potential_next_height = self.height_map[potential_next_pos]
        
        # calculate new sunlight and energy
        self.sunlight = Utils.calculate_sunlight(self.current_day, self.prev_height)
        energy_input = Utils.calculate_energy_input(self.sunlight, self.num_solar_panels, self.daily_panel_input)
        self.battery_level += energy_input
        self.battery_level = min(self.battery_level, self.battery_capacity)
        
        # calculate energy consumption
        energy_consumption = Utils.calculate_energy_consumption(self.dust, self.prev_height, potential_next_height, self.movement_base_energy, self.base_consumption)
        
        # check if enough battery
        if energy_consumption > self.battery_level:
            return False, False
        
        # reduce battery level
        self.battery_level -= energy_consumption
        
        # handle stuck check and movement
        terminated, truncated = self._handle_stuck_and_move(potential_next_pos, potential_next_height, action)

        return terminated, truncated

    def _handle_stuck_and_move(self, potential_next_pos, potential_next_height, action):
        """Handle stuck checks and actual movement"""
        terminated, truncated = False, False
        # check if stuck
        self.is_stuck = Utils.check_stuck(self.dust)
        if self.is_stuck:
            self.stuck_days += 1
            self.step_reward -= 0.1
            # self.reward_after_stuck_penalty = self.total_reward # DEBUG
            if self.stuck_days >= 10:  # perma stuck
                self.step_reward -= 10 # final penalty for perma stuck
                terminated = True
            else:
                pass
        else:
            self.is_stuck = False # not stuck anymore
            self.stuck_days = 0
            
            # check for crash
            if Utils.check_crash(self.prev_height, potential_next_height):
                self.step_reward -= 10  # final crash penalty
                terminated = True
            else:
                # perform movement
                self._perform_move(potential_next_pos, potential_next_height, action)
        
        return terminated, truncated

    def _perform_move(self, new_pos, new_height, action):
        """Perform the actual movement and update state"""
        # update position and height
        self.agent_pos = new_pos
        self.current_height = new_height
        
        # apply height penalty
        height_diff = abs(self.prev_height - self.current_height)
        if height_diff > 0.15:
            height_penalty = min(height_diff * 0.5, 0.125) # cap at -0.125
            self.step_reward -= height_penalty
            # self.reward_after_height_penalty = self.total_reward # DEBUG
        
        # update and apply dust penalty
        self.dust = self.dust_map[self.agent_pos]
        if self.dust > 0.3:
            dust_penalty = min(self.dust * 0.3, 0.15) # cap at -0.15
            self.step_reward -= dust_penalty
            # self.reward_after_dust_penalty = self.total_reward # DEBUG
        
        # apply directional rewards
        self._apply_directional_rewards(action)
        
        # update local values and probabilities
        self._update_local_values()

    def _apply_directional_rewards(self, action):
        """Apply rewards based on movement direction and probabilities"""
        sorted_directions = np.argsort(self.cardinal_probs)[::-1]
        if sorted_directions[0] == action:
            self.step_reward += 0.02  # highest probability direction
            # self.reward_after_1_direction_reward = self.total_reward # DEBUG
        elif sorted_directions[1] == action:
            self.step_reward += 0.01  # second highest probability direction
            # self.reward_after_2_direction_reward = self.total_reward # DEBUG

    def _update_local_values(self):
        """Update local heights, probabilities, and cardinal sums"""
        self.local_heights = self._get_local_values(self.height_map, self.agent_pos)
        self.local_probs = self._get_local_values(self.resource_prob_map, self.agent_pos)
        self.cardinal_probs = self._cardinal_sums()

    def _handle_stay(self):
        """Handle staying in place"""
        # calculate new sunlight level
        self.sunlight = Utils.calculate_sunlight(self.current_day, self.prev_height)

        # calculate energy input (based on sunlight)
        energy_input = Utils.calculate_energy_input(self.sunlight, self.num_solar_panels, self.daily_panel_input)

        # update battery level with energy input
        self.battery_level += energy_input
        self.battery_level = min(self.battery_level, self.battery_capacity)

    def _handle_gather(self):
        """Handle gathering action"""
        # check location before doing anything
        i, j = self.agent_pos
        loc_key = f"{i}_{j}"
        if self.gathered_counts.get(loc_key, 0) >= 4:
            return
    
        # calculate new sunlight level
        self.sunlight = Utils.calculate_sunlight(self.current_day, self.prev_height)

        # calculate energy input (based on sunlight)
        energy_input = Utils.calculate_energy_input(self.sunlight, self.num_solar_panels, self.daily_panel_input)

        # update battery level with energy input
        self.battery_level += energy_input
        self.battery_level = min(self.battery_level, self.battery_capacity)
        
        # calculate energy consumption (base + gathering)
        energy_consumption = self.base_consumption + self.gathering_energy

        # is there enough battery?
        if energy_consumption <= self.battery_level:
            # reduce battery level
            self.battery_level -= energy_consumption

            # process gathering and apply rewards
            self._process_gathering()

    def _process_gathering(self):
        """Process the gathering action and apply rewards"""
        i, j = self.agent_pos

        # location key for tracking
        loc_key = f"{i}_{j}"
        
        # check if location has been gathered before
        previously_gathered = loc_key in self.gathered_counts

        # update gather count
        if not previously_gathered:
            self.gathered_counts[loc_key] = 1

        # # DEBUG
        # print(f"Gathering at {loc_key}")
        # print(f"Previous reward: {self.total_reward}")

        # apply rewards and track resources
        if self.resource_ground_truth[i, j] == 1:
            if not previously_gathered:
                # first time gathering a resource location
                self.step_reward += self.resource_1_reward  # fixed reward for new resource discovery
                self.resources_gathered['resource_1']['count'] += 1
                self.resources_gathered['resource_1']['locations'].add(loc_key)
            else:
                # penalty for repeat gathering
                self.step_reward -= self.repeat_gather_penalty
        
        # update probabilities
        Utils.update_resource_probs(self, i, j)
        self._update_local_values()

        # # DEBUG
        # print(f"New reward: {self.total_reward}")
        # print(f"Reward change: {self.total_reward - self.prev_reward}")

    def _update_observation(self):
        """Update the observation dictionary"""
        # # DEBUG
        # assert abs(self.total_reward) < 500, f"Reward magnitude too large: {self.total_reward}"

        # check if location is depleted
        loc_key = f"{self.agent_pos[0]}_{self.agent_pos[1]}"
        depleted = float(self.gathered_counts.get(loc_key, 0) >= 4)

        self.obs_dict = {
            'battery': np.array([self.battery_level], dtype=np.float32),
            'position': np.array(self.agent_pos, dtype=np.float32),
            'sunlight': np.array([self.sunlight], dtype=np.float32),
            'dust': np.array([self.dust], dtype=np.float32),
            'local_probs': np.array(self.local_probs, dtype=np.float32).flatten(),
            'cardinal_probs': np.array(self.cardinal_probs, dtype=np.float32).flatten(),
            'local_heights': np.array(self.local_heights, dtype=np.float32).flatten(),
            'location_depleted': np.array([depleted], dtype=np.float32)
        }

    @staticmethod
    def _get_height_text_color(height_value, cell_color):
        """Get appropriate text color for height display"""
        # get base color from terrain colormap
        terrain_color = cm.terrain(height_value)
        base_text_color = tuple(int(c * 255) for c in terrain_color[:3])

        # calculate cell brightness
        cell_brightness = sum(cell_color) / 3

        # adjust text color intensity based on cell brightness
        # for dark backgrounds, make text brighter
        if cell_brightness < 128:
            adjusted_color = tuple(min(255, c + 40) for c in base_text_color)
        # for light backgrounds, make text darker
        else:
            adjusted_color = tuple(max(0, c - 40) for c in base_text_color)

        # apply cell brightness factor to maintain readability
        brightness_factor = max(0.3, cell_brightness / 255)  # min brightness of 0.3
        final_color = tuple(int(c * brightness_factor) for c in adjusted_color)

        return final_color
    
    def _get_cell_color(self, i, j, height_value):
        """Determine the color of a cell based on resources, dust, sunlight, and resource probabilities"""
        # start with base gray
        base_color = np.array(self.colors["base_gray"])
        
        # get resource probability for this cell
        prob = self.resource_prob_map[i, j]
        
        # create probability-based color (green tint for high probability)
        prob_color = np.array([100, 200, 100])  # base green color
        prob_intensity = prob * 0.7  # scale down intensity to not overwhelm other visuals
        
        # blend base color with probability color
        base_color = (1 - prob_intensity) * base_color + prob_intensity * prob_color
        
        # check if cell has confirmed resource (from ground truth)
        if self.resource_ground_truth[i, j] == 1:
            base_color = np.array(self.colors["resource"])
        
        # apply dust effect (darker in dusty areas)
        dust_level = self.dust_map[i, j]
        dust_factor = 1 - dust_level * 1.5  # dust darkens the color (1.5 multiplier for stronger effect)
        base_color = base_color * dust_factor
        
        # calculate sunlight effect
        sunlight = Utils.calculate_sunlight(self.current_day, height_value)
        sunlight_factor = 0.3 + sunlight * 0.7  # 0.3 is minimum brightness
        
        # mark gathered locations with red tint
        loc_key = f"{i}_{j}"
        if loc_key in self.gathered_counts:
            # add red tint for gathered locations
            red_color = np.array([255, 50, 50])
            base_color = (0.3 * base_color + 0.7 * red_color)  # fixed blend for gathered locations
        
        # apply sunlight as final factor
        final_color = np.clip(base_color * sunlight_factor, 0, 255).astype(int)

        # ensure final color is within valid range
        final_color = np.clip(final_color, 0, 255)
        
        return tuple(final_color)