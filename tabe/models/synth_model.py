import os
import numpy as np
import random
import torch
import numpy as np

from tabe.utils.logger import logger
from tabe.models.abstractmodel import AbstractModel


# drift_factor :  +/- 10% = [-0.1 ~ 0.1] 
class DrifterModel(AbstractModel):
    def __init__(self, configs, drift_factor=0.1, duration=10, probability=0.1):
        super().__init__(configs, "Drifter") 
        self.drift_factor = drift_factor
        self.duration = duration
        self.probability = probability
        self.drift_occurred = False
        self.counter = 0

    def train(self):
        pass

    def load_saved_model(self):
        pass

    def _forward_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        prev_truth = batch_y[0, -1, -1] 
        pred = prev_truth
        if self.drift_occurred:
            if self.counter < self.duration:
                pred = prev_truth * (1 + self.drift_factor * (1 + 0.1 * ((random.random() - 0.5) * 2))) 
                self.counter += 1
            else:
                self.drift_occurred = False
                self.counter = 0
        elif self.probability > random.random():
            self.drift_occurred = True
        return pred


# noise_factor : (0.0, 0.1)  = 0 ~ 10% of noise
class NoiseModel(AbstractModel):
    def __init__(self, configs, noise_factor=0.05, probability=0.01):
        super().__init__(configs, "Noiser") 
        self.noise_factor = noise_factor
        self.probability = probability

    def train(self):
        pass

    def load_saved_model(self):
        pass

    def _forward_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        prev_truth = batch_y[0, -1, -1] 
        pred = prev_truth 
        if self.probability > random.random():
            pred = prev_truth + self.noise_factor * ((random.random() - 0.5) * 2)
        return pred
    