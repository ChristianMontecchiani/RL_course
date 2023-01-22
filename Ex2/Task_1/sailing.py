# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
import matplotlib as mpl
from collections import namedtuple
from itertools import product
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import io
import imageio

class SailingGridworld(object):
    n_actions = 4
    LEFT, DOWN, RIGHT, UP = range(n_actions)

    def __init__(self, rock_penalty=-2, harbour_reward=10, wind_p=0.1, value_update_iter=100):
        # Array of (state, reward, done, prob) tuples
        self.Transition = namedtuple('Transition', 'state reward done prob')
        self.wind_x = 5, 11
        self.wind_y = 6, 8
        self.rocks1_x = 5, 11
        self.rocks2_x = 5, 11
        self.rocks1_y = 8, 10
        self.rocks2_y = 4, 6
        self.wind_p = wind_p
        self.w = 15
        self.h = 10
        self.wrong_action_prob = 0.05

        self.user_policy = None
        self.user_value_func = None

        self.harbour_x = self.w-1
        self.harbour_y = self.h-1
        self.init_x = 1
        self.init_y = 8

        self.rewards = np.zeros((self.w, self.h))
        self.terminate = np.zeros((self.w, self.h))
        self.state = 0, 0 # print(f"State: {next_state}")
                    # print(f"Reward: {r}")
                    # print(f"Done: {done}")
                    # print(f"Probs: {p}")
        self.fig, self.ax, self.boat = None, None, None
        self.episode_finished = False
        self.transitions = None
        
        # For saving rendered images
        self.value_update_iter = value_update_iter # used as a flag to save gif of value_frames
        self.value_frames, self.env_frames = [], []

        # reset env
        self.reset()

        # Die on rocks, no matter the further action
        self.terminate[self.rocks1_x[0]:self.rocks1_x[1], self.rocks1_y[0]:self.rocks1_y[1]] = 1
        self.terminate[self.rocks2_x[0]:self.rocks2_x[1], self.rocks2_y[0]:self.rocks2_y[1]] = 1
        self.rewards[self.rocks1_x[0]:self.rocks1_x[1], self.rocks1_y[0]:self.rocks1_y[1]] = rock_penalty
        self.rewards[self.rocks2_x[0]:self.rocks2_x[1], self.rocks2_y[0]:self.rocks2_y[1]] = rock_penalty

        # Win on harbour (if you stay)
        self.terminate[self.harbour_x, self.harbour_y] = 1
        self.rewards[self.harbour_x, self.harbour_y] = harbour_reward

        self._update_transitions()


    def is_rocks(self, x, y):
        """Returns True if (x, y) is inside the rocks area, False otherwise"""
        is_rocks1 = self.rocks1_x[0] <= x < self.rocks1_x[1] and self.rocks1_y[0] <= y < self.rocks1_y[1]
        is_rocks2 = self.rocks2_x[0] <= x < self.rocks2_x[1] and self.rocks2_y[0] <= y < self.rocks2_y[1]
        return is_rocks1 or is_rocks2

    def is_wind(self, x, y):
        """Returns True if (x, y) is inside the wind area, False otherwise"""
        is_wind = self.wind_x[0] <= x < self.wind_x[1] and self.wind_y[0] <= y < self.wind_y[1]
        return is_wind

    def _get_next_states_wind(self, state, action):
        """Returns possible state transitions in the windy area.
           We either go in the desired direction, or get carried an extra box in a random direction
           (kinda like being carried back) """
        desired_state = self._get_neighbouring_state(state, action)
        desired_up = self._get_neighbouring_state(desired_state, self.UP)
        desired_down = self._get_neighbouring_state(desired_state, self.DOWN)
        desired_left = self._get_neighbouring_state(desired_state, self.LEFT)
        desired_right = self._get_neighbouring_state(desired_state, self.RIGHT)

        transitions = [self.Transition(desired_state, self.rewards[desired_state[0], desired_state[1]],
                        self.terminate[desired_state[0], desired_state[1]], 1-4*self.wind_p),
                       self.Transition(desired_up, self.rewards[desired_up[0], desired_up[1]],
                        self.terminate[desired_up[0], desired_up[1]], self.wind_p),
                       self.Transition(desired_down, self.rewards[desired_down[0], desired_down[1]],
                        self.terminate[desired_down[0], desired_down[1]], self.wind_p),
                       self.Transition(desired_left, self.rewards[desired_left[0], desired_left[1]],
                        self.terminate[desired_left[0], desired_left[1]], self.wind_p),
                       self.Transition(desired_right, self.rewards[desired_right[0], desired_right[1]],
                        self.terminate[desired_right[0], desired_right[1]], self.wind_p)]

        return transitions

    def _get_possible_transitions(self, state, action):
        """ Overrides the _get_possible_transitions method from the base class
            to account for the wind area"""
        if self.is_wind(*state):
            return self._get_next_states_wind(state, action)
        else:
            return self._get_possible_transitions_standard(state, action)

    def _render(self):
        if self.fig is None:
            self._reset_figure()
        self.update_boat_position()

        io_buf = io.BytesIO()
        self.fig.savefig(io_buf, format='raw', bbox_inches='tight', pad_inches=0)
        io_buf.seek(0)
        img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(self.fig.bbox.bounds[3]), int(self.fig.bbox.bounds[2]), -1))
        io_buf.close()

        return img[:,:,:-1]

    def _reset_figure(self):
        """
        Reset the figure
        :return:
        """
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        xt = np.arange(0, 1, 1/self.w)
        yt = np.arange(0, 1, 1/self.h)
        self.ax.set_xticks(xt)
        self.ax.set_yticks(yt)
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.fig.tight_layout(pad=0)

        # Fill backgroud, add rocks, harbour, and boat
        bg = Rectangle((0, 0), 1, 1, facecolor="#75daff")
        self.ax.add_patch(bg)
        rocks1 = Rectangle((self.rocks1_x[0] / self.w, self.rocks1_y[0] / self.h),
                           (self.rocks1_x[1] - self.rocks1_x[0]) / self.w,
                           (self.rocks1_y[1] - self.rocks1_y[0]) / self.h, facecolor="#c1c1c0")
        self.ax.add_patch(rocks1)
        rocks2 = Rectangle((self.rocks2_x[0] / self.w, self.rocks2_y[0] / self.h),
                           (self.rocks2_x[1] - self.rocks2_x[0]) / self.w,
                           (self.rocks2_y[1] - self.rocks2_y[0]) / self.h, facecolor="#c1c1c0")
        self.ax.add_patch(rocks2)
        wind = Rectangle((self.wind_x[0] / self.w, self.wind_y[0] / self.h),
                         (self.wind_x[1] - self.wind_x[0]) / self.w,
                         (self.wind_y[1] - self.wind_y[0]) / self.h, facecolor="#0F97CA")
        self.ax.add_patch(wind)
        harbour = Rectangle((self.harbour_x / self.w, self.harbour_y / self.h),
                            1 / self.w, 1 / self.h, facecolor="#7AE266")
        self.ax.add_patch(harbour)
        

        if self.state is not None:
            boat_x = np.array([0.1, 0.9, 0.7, 0.3]) / self.w + self.state[0] / self.w
            boat_y = np.array([0.6, 0.6, 0.3, 0.3]) / self.h + self.state[1] / self.h
            self.boat = Polygon(xy=list(zip(boat_x, boat_y)), fill=True,
                           edgecolor="#ac9280", facecolor="#ecc8af")
            self.ax.add_patch(self.boat)
            
        plt.grid(True, color="#e8e8e8", lw=2)

        # Draw everything
        self.fig.canvas.draw()

    def update_boat_position(self):
        if self.state is not None and self.boat is not None:
            boat_x = np.array([0.1, 0.9, 0.7, 0.3]) / self.w + self.state[0] / self.w
            boat_y = np.array([0.6, 0.6, 0.3, 0.3]) / self.h + self.state[1] / self.h
            self.boat.set_xy(list(zip(boat_x, boat_y)))


    def _update_transitions(self):
        """Updates the state transition model after rewards etc. were changed."""
        self.transitions = np.empty((self.w, self.h, self.n_actions), dtype=list)

        for x, y, a in product(range(self.w), range(self.h), range(self.n_actions)):
            self.transitions[x, y, a] = self._get_possible_transitions((x, y), a)

    def reset(self):
        """ Resets the environment to the initial state
        Returns:
            The initial state of the environment."""
        if self.fig:
            plt.close(self.fig)
        
        self.state = self.init_x, self.init_y
        self._reset_figure()
        self.episode_finished = False

        # clean the frame buffer
        self.value_frames, self.env_frames = [], []

        return self.state

    def _get_neighbouring_state(self, state, relative_pos):
        """Returns the next state to be reached when action is taken in state.
           Assumes everything to be deterministic.
           Args:
               state: current state
               relative_pos: action to be taken/evaluated
            Returns:
                The next state (as numpy.array)"""
        if relative_pos == self.LEFT:
            if state[0] > 0:
                return state[0]-1, state[1]
            else:
                return state
        elif relative_pos == self.RIGHT:
            if state[0] < self.w-1:
                return state[0]+1, state[1]
            else:
                return state
        elif relative_pos == self.DOWN:
            if state[1] > 0:
                return state[0], state[1]-1
            else:
                return state
        elif relative_pos == self.UP:
            if state[1] < self.h-1:
                return state[0], state[1]+1
            else:
                return state
        else:
            raise ValueError("Invalid action: %s" % relative_pos)

    def _get_possible_transitions_standard(self, state, action):
        """ Returns an array of possible future states when
            given action is taken in given state.
            Args:
                state - current state
                action -  action to be taken/evaluated
            Returns:
                 an array of (state, reward, done, prob) uples:
                [(state1, reward1, done1, prob1), (state2, reward2, done2, prob2)...].
                State is None if the episode terminates."""
        if self.terminate[state[0], state[1]]:
            return [self.Transition(None, 0, True, 1)]
        transitions = []
        action1 = (action-1) % self.n_actions
        state1 = self._get_neighbouring_state(state, action1)
        reward1 = self.rewards[state1[0], state1[1]]
        terminate1 = self.terminate[state1[0], state1[1]]
        transitions.append(self.Transition(state1, reward1, terminate1, self.wrong_action_prob))

        action2 = (action + 1) % self.n_actions
        state2 = self._get_neighbouring_state(state, action2)
        reward2 = self.rewards[state2[0], state2[1]]
        terminate2 = self.terminate[state2[0], state2[1]]
        transitions.append(self.Transition(state2, reward2, terminate2, self.wrong_action_prob))

        state3 = self._get_neighbouring_state(state, action)
        reward3 = self.rewards[state3[0], state3[1]]
        terminate3 = self.terminate[state3[0], state3[1]]
        transitions.append(self.Transition(state3, reward3, terminate3, 1-2*self.wrong_action_prob))
        return transitions

    def step(self, action):
        """ Moves the simulation one step forward.
        Args:
            action: The action taken by the agent (int)
        Returns:
            Tuple (new_state, reward, done, info)
            new_state: new state of the environment
            reward: reward for the transition
            done: whether the environment is finished or not
            info: empty dictionary """
        if self.episode_finished:
            print("Episode is finished! Reset the environment first!")
            return self.state, 0, True, {}
        info = {}

        # Get possible next states for this action, along with their probabilities
        action = int(action)
        transitions = self.transitions[self.state[0], self.state[1], action]

        # Sample next state from the transitions.
        r = np.random.rand()
        for state, reward, done, p in transitions:
            if r < p:
                self.state = state
                break
            else:
                r -= p

        self.episode_finished = done

        return self.state, reward, done, info

    def draw_values_policy(self, values, policy):
        self.clear_text()
        # draw value
        self.user_value_func = values
        self._draw_floats(values, v_offset=0.5, label="V")
        self._draw_floats(self.rewards, v_offset=0.8, label="r")

        # draw actions
        self.user_policy = policy
        pol_str = policy.astype(int).astype(str)
        pol_str[pol_str == str(self.RIGHT)] = "Right"
        pol_str[pol_str == str(self.LEFT)] = "Left"
        pol_str[pol_str == str(self.UP)] = "Up"
        pol_str[pol_str == str(self.DOWN)] = "Down"
        for i, row in enumerate(pol_str):
            rx = (i+0.5)/self.w
            for j, value in enumerate(row):
                ry = (j+0.2)/self.h
                self.ax.text(rx, ry, "a: {}".format(value), ha='center', va='center', fontsize="medium")

        # for visualization
        self.value_frames.append(self._render())
        
        # after updating, save the frames to gif for visualization
        if len(self.value_frames) >= self.value_update_iter:
            frames = np.stack(self.value_frames)
            imageio.mimsave('value.gif', frames, format = 'GIF-PIL', fps = 1)

    
    # render env
    def render(self,):
        img = self._render()
        self.env_frames.append(img)
        # save to the env.gif for visualization
        if self.episode_finished: 
            frames = np.stack(self.env_frames)
            imageio.mimsave('env.gif', frames, format = 'GIF-PIL', fps = 5) 


    def _draw_floats(self, values, v_offset=0.8, label="V"):
        """Draw an array of float values on the grid.
           Doesn't automatically render the environment - a separate call
           to render is needed afterwards.
           Args:
               values: a width*height array of floating point numbers"""
        for i, row in enumerate(values):
            rx = (i+0.5)/self.w
            for j, value in enumerate(row):
                ry = (j+v_offset)/self.h
                self.ax.text(rx, ry, "{}={:.2f}".format(label, value), ha='center', va='center', fontsize="medium")

    def clear_text(self):
        """Removes all text from the environment before it's rendered."""
        self.ax.texts.clear()