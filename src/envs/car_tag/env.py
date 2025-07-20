import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class CarTagEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        # Constants
        self.pursuer_speed = 1.0  # s1
        self.evader_speed = 0.5   # s2
        self.pursuer_turn_radius = 1.0  # R

        self.window_size = 600
        self.scale = 30  # 1 unit = 30 pixels

        self.screen = None
        self.clock = None

        self.pursuer_trail = []
        self.evader_trail = []

        # Define the action space for the pursuer (phi)
        # Continuous value between -1 and 1, representing a turn rate ratio
        pursuer_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define the action space for the evader (psi)
        # Continuous value representing a heading angle in radians (-pi to pi)
        evader_action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)

        # Combine the action spaces for both agents into a dictionary
        self.action_space = spaces.Dict({
            "pursuer_action": pursuer_action_space,
            "evader_action": evader_action_space,
        })

        # (Observation space and other environment details would also be defined here)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        ) # Example based on state parameters [12, 13]

        self.state = None

    def step(self, action):
        # Process actions from the dictionary
        phi = action["pursuer_action"][0]
        psi = action["evader_action"][0]

        xp, yp, theta, xe, ye = self.state

        # Compute theta_dot for the pursuer
        theta_dot = (self.pursuer_speed / self.pursuer_turn_radius) * phi

        # Update pursuer position and heading
        theta_new = theta + theta_dot
        xp_new = xp + self.pursuer_speed * np.sin(theta_new)
        yp_new = yp + self.pursuer_speed * np.cos(theta_new)

        # Update evader position
        xe_new = xe + self.evader_speed * np.sin(psi)
        ye_new = ye + self.evader_speed * np.cos(psi)

        # Update state
        self.state = np.array([xp_new, yp_new, theta_new, xe_new, ye_new], dtype=np.float32)

        # Compute reward and termination condition (can be customized)
        distance = np.linalg.norm([xp_new - xe_new, yp_new - ye_new])
        reward = -distance  # reward for getting closer
        terminated = distance < 0.5  # tag condition
        truncated = False

        info = {}
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # (Reset environment state)
        self.state = self.observation_space.sample()  # assign initial state
        self.pursuer_trail = []
        self.evader_trail = []
        observation = self.state
        info = {}
        return observation, info

    def render(self):
        if self.render_mode != "rgb_array":
            raise ValueError("Only 'rgb_array' render_mode is supported for video recording.")

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        xp, yp, theta, xe, ye = self.state

        def world_to_screen(x, y):
            return int(self.window_size / 2 + x * self.scale), int(self.window_size / 2 - y * self.scale)

        pursuer_pos = world_to_screen(xp, yp)
        evader_pos = world_to_screen(xe, ye)

        self.pursuer_trail.append(pursuer_pos)
        self.evader_trail.append(evader_pos)

        if len(self.pursuer_trail) > 1:
            pygame.draw.lines(self.screen, (255, 0, 0), False, self.pursuer_trail, 2)
        if len(self.evader_trail) > 1:
            pygame.draw.lines(self.screen, (0, 0, 255), False, self.evader_trail, 2)

        pygame.draw.circle(self.screen, (255, 0, 0), pursuer_pos, 6)
        heading_x = int(pursuer_pos[0] + 15 * np.sin(theta))
        heading_y = int(pursuer_pos[1] - 15 * np.cos(theta))
        pygame.draw.line(self.screen, (255, 0, 0), pursuer_pos, (heading_x, heading_y), 2)

        pygame.draw.circle(self.screen, (0, 0, 255), evader_pos, 6)

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
            self.clock = None
