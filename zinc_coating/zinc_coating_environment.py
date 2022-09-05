import gym
from gym import spaces

import numpy as np

from zinc_coating.base import ZincCoatingBase as base


class ZincCoatingV0(gym.Env):
    """Simple continuous zinc coating environment"""

    def __init__(self, steps_per_episode=2500, coating_reward_time_offset=0, append_is=False, random_coating_targets=True, random_coil_characteristics=True, random_coil_lengths=True, random_coil_speed=True, stable_speed=True):
        super(ZincCoatingV0, self).__init__()
        ## Episodes only end when number of steps is reached
        ## 2500 ensures that every steel configuration is processed at least twice, hence each episode provides all information necessary
        self.steps_per_episode = steps_per_episode

        ## Observation delay = Action delay
        self.coating_reward_time_offset = coating_reward_time_offset
        self.action_buffer = []
        for _ in range (coating_reward_time_offset):
            self.action_buffer.append(0)

        ## State space with state augmention (is = information state)
        self.append_is = append_is
        self.observation_dim = 13
        if self.append_is:
            self.observation_dim += coating_reward_time_offset

        self.action_space = spaces.Box(
            np.array([0]), np.array([1]), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.observation_dim,), dtype=np.float32)

        self.base = base(
            coating_reward_time_offset=coating_reward_time_offset,
            random_coating_targets=random_coating_targets,
            random_coil_characteristics=random_coil_characteristics,
            random_coil_lengths=random_coil_lengths,
            random_coil_speed=random_coil_speed,
            stable_speed=stable_speed
        )
        self.seed()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 10000000)
        self.base.seed(seed)
        return [seed]

    def step(self, nozzle_pressure):
        self.current_step += 1
        if (self.current_step >= self.steps_per_episode):
            self.done = True
        else:
            self.done = False

        observation, reward, zinc_coating_real = self.base.step(nozzle_pressure[0]*700)
        if self.coating_reward_time_offset > 0:
            self.action_buffer.pop(0)
            self.action_buffer.append(nozzle_pressure[0])

        if self.append_is:
            return np.asarray(self._transform_observation(observation) + tuple(self.action_buffer)), reward, self.done, {"coating": zinc_coating_real}

        return np.asarray(self._transform_observation(observation)), reward, self.done, {"coating": zinc_coating_real}

    def reset(self):
        self.current_step = 0
        observation, _, _ = self.base.reset()

        self.action_buffer = []
        for _ in range (self.coating_reward_time_offset):
            self.action_buffer.append(0)

        if self.append_is:
            return np.asarray(self._transform_observation(observation) + tuple(self.action_buffer))
        
        return np.asarray(self._transform_observation(observation))

    def render(self, mode='human', close=False):
        print("hey")

    def _transform_observation(self, observation):
        coating_delta = observation.zinc_coating - observation.current_coating_target
        return ((observation.current_coating_target - 8) / 202,
                (observation.zinc_coating - 8) / 202,
                (observation.nozzle_pressure / 700),
                (coating_delta + 50) / 220,
                (1 if coating_delta < 0 else 0),
                (1 if coating_delta >= 0 and coating_delta <= 20 else 0),
                (1 if coating_delta > 20 else 0)) + one_hot_encode(observation.next_coil_type if observation.coil_switch_next_tick else observation.current_coil_type, 6)


def one_hot_encode(to_encode, discrete_states):
    output = [0] * discrete_states
    output[to_encode] = 1
    return tuple(output)
