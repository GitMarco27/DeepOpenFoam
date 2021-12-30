import logging
import math
import random

import gym
from IPython.core.display import clear_output
from gym.spaces import Box, Discrete
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .env_models_functions import pred_global_variables, decode

# This file contains all classes based on environment based on latent parameters generated by an autoencoder.
# The LDEnv class describes the operations that all subclasses derive from it
from . import reward_functions
from inspect import getmembers, isfunction

def get_all_rewards_functions():
    list_reward_fucntions_with_keys = getmembers(reward_functions, isfunction)

    possible_reward_functions = {}
    for element in list_reward_fucntions_with_keys:
        possible_reward_functions[element[0]] = element[1]
    return possible_reward_functions


class LDEnv(gym.Env):
    # starting geometry index
    current_index = 0

    # max number of steps per episode
    num_steps_max = None

    current_step = 0

    # The state is composed of the latent parameters and other global parameters
    _state = None

    # boolenan variable that tells if the episode is concluded
    _episode_ended = False

    # I also save the current geometry that I use later in the render
    _current_geom = 1

    # save the old action
    _old_action = None

    # Size of observation and action spaces
    action_space = None
    observation_space = None

    ref_value = 10000

    def __init__(self, models, data_env, rl_config):
        super(LDEnv, self).__init__()
        # Load autoencoders and prediction models.
        self.models = models


        # I get the environment variables from the configuration file
        self.num_steps_max = rl_config['steps_per_episode']

        # In date env we find several dait:
        # 1. Latent data
        # 2. Original data
        self.data_env = data_env

        possible_reward_functions = get_all_rewards_functions()
        self.get_reward = possible_reward_functions[rl_config['type_rewards']]

        # get number of latent parameters
        self._number_of_latent_parameters = self.data_env['cod'].shape[1]

        self._number_of_global_variables = rl_config['number_of_global_variables']

        # get the max value of action
        self.delta_value = rl_config['delta_value']

        # Calculates parameters for normalization as maximum and minimum value for each of latents and then adds them
        # into data_env dict
        min_values_latent, max_values_latent = self.calc_min_max_of_state(self._number_of_global_variables)

        self.data_env['min_values_latent'] = min_values_latent
        self.data_env['max_values_latent'] = max_values_latent

        # I reset the status that will have a size equal to the number of latent
        # parameters plus the value of the global quantity
        self._state = [0] * (self._number_of_latent_parameters + self._number_of_global_variables)

        # Observation space
        min_value_observation = np.array(self.data_env['min_values_latent'].tolist()+[-5]*self._number_of_global_variables)
        max_value_observation = np.array(self.data_env['max_values_latent'].tolist()+[5]*self._number_of_global_variables)

        self.observation_space = Box(low=min_value_observation, high=max_value_observation, dtype=np.float64)

    # Function used to calculate the maximum and minimum for each of the latent params
    def calc_min_max_of_state(self, number_of_global_variables):
        pd_all_cod = pd.DataFrame(self.data_env['cod'])
        min_latent = pd_all_cod.min().to_numpy()
        max_latent = pd_all_cod.max().to_numpy()
        return min_latent.reshape(-1) , max_latent.reshape(-1)

    def _get_obs(self):
        return np.array(self._state)

    def get_latent_data_from_state(self):
        return self._get_obs()[:self._number_of_latent_parameters]

    def get_global_variable_from_state(self):
        return self._get_obs()[self._number_of_latent_parameters:]

    def calculate_global_variabls(self, current_laten_params):
        self._current_geom = decode(self.models['decoder'], current_laten_params)

        global_variables = pred_global_variables(current_laten_params, self.models)

        return global_variables

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self._episode_ended = False

        # I randomly choose a new geometry for this episode
        self.current_index = random.randint(0, self.data_env['cod'].shape[0] - 1)

        # Extract current latents
        current_laten_params = self.data_env['cod'][self.current_index]

        self._state = current_laten_params.reshape(-1).tolist()

        # Calculate global variables
        global_variables = self.calculate_global_variabls(current_laten_params)

        self._state.extend(global_variables)

        self.num_step = 0

        Cl = self._state[-2]
        Cd = self._state[-1]

        self.ref_value = Cl / Cd

        return self._get_obs()

    def update_latent(self, current_laten_params, action):
        return None

    def step(self, action):
        self.num_step += 1

        if self._episode_ended:
            return self.reset()

        # I sum the current latent parameters with actions
        current_laten_params = self.get_latent_data_from_state()

        new_latent_params = self.update_latent(current_laten_params, action)

        self._state = new_latent_params.reshape(-1).tolist()

        # calculate global variables and add them to the state
        global_variables = self.calculate_global_variabls(current_laten_params)

        for variable in global_variables:
            self._state.append(variable)

        # update best_value obtained
        # if self._state[-1] > self.best_value_obtained:
        #     self.best_value_obtained = self._state[-1]

        # Check if the episode is finished
        if self.num_step >= self.num_steps_max:
            self._episode_ended = True

        # calculate the reward
        reward = self.get_reward(self._state, self.ref_value, self._episode_ended)

        # current value
        Cl = self._state[-2]
        Cd = self._state[-1]

        current_value = Cl / Cd

        return self._get_obs(), reward, self._episode_ended, {'best_value_obtained': self.ref_value,
                                                              'current_value': current_value}

    def render(self):
        clear_output(wait=True)

        starting_value = round(self.ref_value, 4)

        Cl = self._state[-2]
        Cd = self._state[-1]

        current_value = Cl / Cd

        current_value = round(current_value, 4)
        delta = 100 * round(current_value - starting_value, 4)

        if delta < 0:
            color_points = 'red'
        elif delta > 0:
            color_points = 'green'
        else:
            color_points = 'blue'

        colors = np.array(['green'] * self._old_action.shape[0])
        colors[self._old_action < 0] = 'red'
        colors[self._old_action == 0] = 'blue'

        fig, ax = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(
            f'Env: step {self.current_step}, current_value {current_value}, starting eff : {starting_value}, '
            f'Delta eff: {delta}', fontsize=16)

        ax[0, 0].scatter(self.data_env['origin_geom'][self.current_index][:, 0],
                         self.data_env['origin_geom'][self.current_index][:, 1], c=color_points, marker='o')
        ax[0, 0].set_title('Starting Geom')
        ax[0, 1].scatter(self._current_geom[0][:, 0], self._current_geom[0][:, 1], c='blue', marker='o')
        ax[0, 1].set_title('Current Geom')



        ax[1, 0].scatter(np.arange(len(self._state[:self._number_of_latent_parameters])),
                         self._state[:self._number_of_latent_parameters], c=colors)
        ax[1, 0].set_title('Current latent parameters')
        # ax[1,0].title('latent_params')

        ax[1, 1].scatter(np.arange(len(self._old_action)), self._old_action, c=colors)
        ax[1, 1].set_ylim([-self.delta_value, self.delta_value])
        ax[1, 1].set_title('Old action')
        plt.show()


class DiscreteActionLDEnv(LDEnv):
    def __init__(self, models, data_env, rl_config):
        super(DiscreteActionLDEnv, self).__init__(models, data_env, rl_config)

        # the actions are divided in this way
        # 0 I do nothing
        # 1-> Number of latent parameters: I add to the latent amount
        # Number of latent parameters+1-> End:  subtract from the latent equivalent
        self.action_space = Discrete(2 * rl_config['delta_value'] * self._number_of_latent_parameters + 1)


    def update_latent(self, current_laten_params, action):
        if action == 0:
            return current_laten_params

        value_to_sum = math.floor(action / 2048) + 1

        if action > 2 * self._number_of_latent_parameters + 1:
            action = action - ((2 * self._number_of_latent_parameters) * (value_to_sum - 1))

        self._old_action = np.zeros(current_laten_params.shape)

        if action > self._number_of_latent_parameters:
            self._old_action[int(action - 1024 - 1)] = -value_to_sum
            current_laten_params[int(action - 1024 - 1)] -= (value_to_sum / 100) * current_laten_params[
                int(action - 1024 - 1)]
        elif action > 0:
            self._old_action[int(action - 1)] = value_to_sum
            current_laten_params[int(action - 1)] += (value_to_sum / 100) * current_laten_params[int(action - 1)]

        new_latent_params = current_laten_params
        return new_latent_params


class BoxActionLDEnv(LDEnv):
    def __init__(self, models, data_env, rl_config):
        super(BoxActionLDEnv, self).__init__(models, data_env, rl_config)

        delta_action = (self.delta_value/100)*abs(self.data_env['max_values_latent']-self.data_env['min_values_latent'])
        # Define action and observation space
        # They must be gym.spaces objects
        # self.action_space = spaces.Box(low=-np.array(1024), high=np.array(1024), dtype=np.int64)
        self.action_space = Box(low=-delta_action, high=delta_action,
                                dtype=np.float32)

    def update_latent(self, current_laten_params, action):
        self._old_action = action
        current_laten_params += action
        return current_laten_params
