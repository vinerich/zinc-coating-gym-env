from zinc_coating.coil import Coil
from zinc_coating.nozzle import Nozzle
from zinc_coating.zinc_bath import ZincBath
from zinc_coating.observation import Observation
import numpy as np
import queue


class Constants:
    TIMESTEP = 100  # ms


class ZincCoatingBase():
    def __init__(self, coating_reward_time_offset=0, random_coating_targets=True, random_coil_characteristics=True, random_coil_lengths=True, random_coil_speed=True, stable_speed=False):
        self.use_randomized_coil_targets = random_coating_targets
        self.use_randomized_coil_characteristics = random_coil_characteristics
        self.use_changing_coil_speed = random_coil_speed
        self.use_randomized_coil_lengths = random_coil_lengths
        self.coating_reward_time_offset = np.max(
            [0, coating_reward_time_offset])
        self.delay_queue = queue.Queue()

        self.nozzle = Nozzle(np.random.randint(0, 700))
        self.zinc_bath = ZincBath()

        self.stable_speed = stable_speed
        self.num_stable_speed = 0

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 10000000)
        np.random.seed(seed)

        print("Seed to use: ", seed)

    def reset(self):
        self.timestep = 0
        self.coil_speed = 160 / 60  # 2,667
        self.coil_types = [0] * 6

        if(self.use_changing_coil_speed):
            self.coil_speed = np.random.randint(80, 200) / 60  # [1,333, 3,333]
            self.coil_speed_target = np.random.randint(80, 200) / 60  # [1,333, 3,333]
        else:
            self.coil_speed_target = self.coil_speed

        self.delay_queue = queue.Queue()
        for _ in range(self.coating_reward_time_offset):
            self.delay_queue.put((0, 0))

        self.nozzle = Nozzle(np.random.randint(0, 700))
        self.zinc_bath = ZincBath()
        self.zinc_bath.step()

        self.current_coil = self.getNewCoil()
        self.current_coil.start(self.timestep, self.coil_speed)
        self.next_coil = self.getNewCoil()

        zinc_bath_coating = self.zinc_bath.getZincCoatingForCoil(
            self.current_coil.getZincCoatingCharacteristic(), self.coil_speed)
        zinc_coating = self.nozzle.getZincCoating(
            zinc_bath_coating, self.coil_speed)

        return Observation(self.timestep, self.coil_speed, self.current_coil.type, self.current_coil.getZincCoatingTarget(), False, self.next_coil.type, self.next_coil.getZincCoatingTarget(), self.current_coil.max_length, zinc_bath_coating, 0, self.nozzle.getPressure()), 0, 0

    def step(self, new_pressure):
        self.timestep += Constants.TIMESTEP
        self.nozzle.setPressure(new_pressure)
        self.zinc_bath.step()

        # print(f"coil_speed {self.coil_speed}")
        if(self.use_changing_coil_speed):
            if not (np.absolute(self.coil_speed - self.coil_speed_target) < 0.1):
                if(self.coil_speed_target < self.coil_speed):
                    self.coil_speed -= 0.5 / 1000 * Constants.TIMESTEP
                else:
                    self.coil_speed += 0.5 / 1000 * Constants.TIMESTEP
            else:
                self.num_stable_speed += 1
                if self.stable_speed:
                    if self.num_stable_speed >= 400:
                        self.num_stable_speed = 0
                        self.coil_speed_target = np.random.randint(80, 200) / 60
                else:
                    self.coil_speed_target = np.random.randint(80, 200) / 60

        coil_length, coil_switch_next_tick = self.current_coil.getLength(
            self.timestep, self.coil_speed)
        if(coil_length < 0):
            self.current_coil = self.next_coil
            self.current_coil.start(self.timestep, self.coil_speed)
            self.next_coil = self.getNewCoil()

            coil_length = self.current_coil.max_length
            coil_switch_next_tick = False

        zinc_bath_coating = self.zinc_bath.getZincCoatingForCoil(
            self.current_coil.getZincCoatingCharacteristic(), self.coil_speed)
        zinc_coating = self.nozzle.getZincCoating(
            zinc_bath_coating, self.coil_speed)

        zinc_coating_diff = zinc_coating - self.current_coil.getZincCoatingTarget()
        
        self.delay_queue.put((self._get_reward(zinc_coating_diff), zinc_coating))

        current_reward, current_zinc_coating = self.delay_queue.get()
        return Observation(self.timestep, self.coil_speed, self.current_coil.type, self.current_coil.getZincCoatingTarget(), coil_switch_next_tick, self.next_coil.type, self.next_coil.getZincCoatingTarget(), coil_length, zinc_bath_coating, current_zinc_coating, self.nozzle.getPressure()), current_reward, current_zinc_coating

    def getNewCoil(self):
        coil_type = 0
        if(self.use_randomized_coil_targets or self.use_randomized_coil_characteristics):
            if(np.max(self.coil_types) == 0):
                coil_type = np.random.randint(6)
            else:
                probabilities = np.max(self.coil_types) - self.coil_types
                if(np.sum(probabilities) == 0):
                    probabilities = [1/6] * 6
                else:
                    probabilities = probabilities / np.sum(probabilities)

                coil_type = np.random.choice(6, p=probabilities)

        coil_length = 100
        if(self.use_randomized_coil_lengths):
            coil_length = np.random.randint(50, 150)

        self.coil_types[coil_type] += 1
        return Coil(type=coil_type, length=coil_length, rand_target=self.use_randomized_coil_targets, rand_characteristic=self.use_randomized_coil_characteristics)

    def _get_reward(self, zinc_coating_diff):
        if zinc_coating_diff < 0:
            reward = -2
        elif zinc_coating_diff > 20:
            reward = -1
        else:
            reward = 1 / (zinc_coating_diff + 1)

        return reward
