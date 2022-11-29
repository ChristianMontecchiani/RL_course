"""
A 2D reacher environment.
Adapted from the OpenAI Gym Acrobot environment developed by Christoph Dann
and released under the 3-clause BSD license.
"""
#%%
from typing import Optional
from zoneinfo import reset_tzpath
import numpy as np
from numpy import sin, cos, pi
from gym import core, spaces
from gym.utils import seeding
from gym.envs.registration import register
from gym.envs.classic_control import utils
from gym.utils.renderer import Renderer
import ipdb


class ReacherEnv(core.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 15,
        }
    SCREEN_DIM = 500

    def __init__(self, render_mode: Optional[str] = None, max_episode_steps=200):
        high = np.ones(2) * np.inf
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.state = None
        self.goal = np.array([1.0, 1.0])
        self.termination_threshold = 0.25
        self.seed()
        self.link_length_1 = 1.
        self.link_length_2 = 1.
        self.prev_cartesian_pos = np.zeros(2)
        self.prev_state = np.zeros(2)
        self.step_angle_change = 0.2
        self.substeps = 10

        self.clock = None
        self.isopen = True
        self.screen = None
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render_frame)

        self.max_episode_steps = max_episode_steps
        self._counter = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(2,))
        #self.state = np.array([-3.14, -1.57])

        self.renderer.reset()
        self.renderer.render_step()

        self._counter = 0
        return self.state

   
  

    """
    def get_reward(self, prev_state, action, next_state):
        # TODO: Task 3: Implement and test two reward functions
        ########## Your code starts here ##########
        
        prev_pos = self.get_cartesian_pos(prev_state)
        prev_dist = np.sqrt(np.sum(np.square(self.goal - prev_pos))) # Euclidean distance 
        
        return -prev_dist

        ########## Your codes end here ########## 
    
    """ 
    def get_reward(self, prev_state, action, next_state):
        prev_theta_0 = prev_state[0]
        next_theta_0 = next_state[0]

        prev_theta_1 = prev_state[1]
        next_theta_1 = next_state[1]


        if prev_theta_0 == next_theta_0 and prev_theta_1 != next_theta_1:
            return 1 
        
        return 0

        ########## Your codes end here ########## 
         


    def get_cartesian_pos(self, state):
        ee_pos = np.zeros(2)
        ee_pos[0] = np.sin(state[0])*self.link_length_1 + \
                np.sin(state[0]+state[1])*self.link_length_2
        ee_pos[1] = -np.cos(state[0])*self.link_length_1 - \
                np.cos(state[0]+state[1])*self.link_length_2
        return ee_pos

    @property
    def cartesian_pos(self):
        return self.get_cartesian_pos(self.state)

    def step(self, a):
        self._counter += 1
        self.prev_cartesian_pos = self.cartesian_pos
        self.prev_state = np.copy(self.state)
        dpos = self.step_angle_change / self.substeps
        joint = a//2
        dpos = dpos*(-1)**a

        # Do the simulation in substeps to avoid a situation where we jump to
        # the other side without terminating the episode
        for _ in range(self.substeps):
            if a < 4:
                self.state[joint] += dpos

        terminal = self.get_terminal_state()
        truncked = self._counter >= self.max_episode_steps # truck the timesteps
        terminal |= truncked

        # Compute the reward
        reward = self.get_reward(self.prev_state, a, self.state)
        
        self.renderer.render_step()

        return (self.state, reward, terminal, {})

    def get_terminal_state(self):
        terminal_distance = np.sqrt(np.sum((self.cartesian_pos - self.goal)**2))
        terminal = terminal_distance < self.termination_threshold
        return terminal

    def render(self):
        return self.renderer.get_renders() 
    
    def _render_frame(self, mode='human'):
        self.metadata["render_modes"]

        assert mode in self.metadata['render_modes']
        
        import pygame
        from pygame import gfxdraw
      

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        s = self.state

        bound = self.link_length_1 + self.link_length_2 + 0.2  # 2.2 for default
        scale = (self.SCREEN_DIM / (bound * 2))/2
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        p1 = [-self.link_length_1 *cos(s[0]) * scale, 
            self.link_length_1 * sin(s[0]) * scale]

        p2 = [p1[0] - self.link_length_2 * cos(s[0] + s[1]) * scale,
              p1[1] + self.link_length_2 * sin(s[0] + s[1]) * scale]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.link_length_1 * scale, self.link_length_2 * scale]


        # draw links
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            x += offset
            y += offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1*scale
            coords = [(l, b), (l, t), (r, t), (r, b)]

            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)

            gfxdraw.aapolygon(surf, transformed_coords, (73, 73, 71))
            gfxdraw.filled_polygon(surf, transformed_coords, (73, 73, 71))

            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (73, 73, 71))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (73, 73, 71))

        # draw the end-effector pos
        end_effector = p2[::-1]

        end_effector = (end_effector[0] + offset, end_effector[1] + offset)
        gfxdraw.aacircle(surf, int(end_effector[0]), int(end_effector[1]), int(0.1*scale), (16, 212, 108))
        gfxdraw.filled_circle(surf, int(end_effector[0]), int(end_effector[1]), int(0.1*scale), (16, 212, 108))

        # draw the goal pos
        goal_pos = (self.goal[0] * scale + offset, self.goal[1] * scale + offset)
        gfxdraw.aacircle(surf, int(goal_pos[0]), int(goal_pos[1]), int(0.1*scale), (235, 84, 97))
        gfxdraw.filled_circle(surf, int(goal_pos[0]), int(goal_pos[1]), int(0.1*scale), (235, 84, 97))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

register("Reacher-v1",
        entry_point="%s:ReacherEnv"%__name__,
        max_episode_steps=200)
