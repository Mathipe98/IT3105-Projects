from typing import Any, Dict, Union, List
import numpy as np
from itertools import product
from pprint import pprint

from tile_coding import create_tilings, get_tile_coding


class CartPoleGame:

    def __init__(self,
                 g: float=-9.81,
                 m_c: float=1,
                 m_p: float=1,
                 l: float=0.5,
                 tau: float=0.02,
                 n_tilings: int=2,
                 feat_bins: List=None) -> None:
        if feat_bins is None:
            feat_bins = [4, 4, 8, 8, 2]
        self.g = g
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.tau = tau
        self.n_tilings = n_tilings
        self.feat_bins = feat_bins
        # Create vectors for positions and velocities (and their derivatives) with length equal to the total number of steps
        T = 300
        steps = int(np.ceil(T / tau))
        self.x = np.zeros(steps)
        self.dx = np.zeros(steps)
        self.d2x = np.zeros(steps)
        self.theta = np.zeros(steps)
        self.dtheta = np.zeros(steps)
        self.d2theta = np.zeros(steps)

    def generate_encoding(self) -> np.ndarray:
        """This method generates the tile-encoding for the cart problem.

        Returns:
            np.ndarray: Array that is used for indexing and encoding the features
                        in the problem (i.e. positions and velocities + time)
        """
        x_range = [-2.4, 2.4]
        x_v_range = [-1, 1]
        ang_range = [-0.21, 0.21]
        ang_v_range = [-0.1, 0.1]
        t_range = [0, 300]
        ranges = [x_range, x_v_range, ang_range, ang_v_range, t_range]
        # Create a nested list that uses the same bins for every tiling
        bins = [bin for bin in self.feat_bins]
        bins = [bins[:] for _ in range(self.n_tilings)]
        offset = 0
        offset_list = []
        for _ in range(self.n_tilings):
            current = []
            for i in range(len(ranges) - 1):
                a = ranges[i][0]
                b = ranges[i][1]
                ab_sum = abs(a) + abs(b)
                # Let the offset for a particular feature be 20% of the feature itself
                # Then double that every iteration with offset variable
                feat_offset = round(ab_sum * 0.2 * offset, 4)
                current.append(feat_offset)
            # Append 150 last because time offset is constant
            current.append(150)
            offset_list.append(current)
            offset += 1
        tilings = create_tilings(ranges, self.n_tilings, bins, offset_list)
        return tilings

    def get_next_state_parameters(self, current_parameters: List, action: float) -> List:
        elapsed_time = current_parameters[-1]
        print(f"Elapsed time: {elapsed_time}")
        t = int(elapsed_time / self.tau) + 1
        print(f"t value: {t}")
        self.theta[t] = self.theta[t-1] + self.tau * self.dtheta[t-1]
        self.dtheta[t] = self.theta[t-1] + self.tau * self.d2theta[t-1]
        self.d2theta[t] = self.compute_d2theta(action, t)
        self.x[t] = self.x[t-1] + self.tau * self.dx[t-1]
        self.dx[t] = self.dx[t-1] + self.tau * self.d2x[t-1]
        self.d2x[t] = self.compute_d2x(action, t)
        next_state_parameters = [
            self.x[t],
            self.dx[t],
            self.theta[t],
            self.dtheta[t],
            elapsed_time + self.tau
        ]
        return next_state_parameters

    def compute_d2x(self, F: float, t: int) -> float:
        var_1 = self.m_p * self.l * (self.dtheta[t]**2 * np.sin(self.theta[t]) - self.d2theta[t] * np.cos(self.theta[t]))
        var_2 = self.m_c + self.m_p
        result = (F + var_1) / var_2
        return result
    
    def compute_d2theta(self, F: float, t: int) -> float:
        denom_part = (self.m_p * self.theta[t] ** 2) / (self.m_c + self.m_p)
        denominator = self.l * (4/3 - denom_part)
        numer_part = (-F - self.m_p * self.l * self.dtheta[t]**2 * np.sin(self.theta[t])) / (self.m_c + self.m_p)
        numerator = (self.g * np.sin(self.theta[t]) + np.cos(self.theta[t]) * numer_part)
        result = numerator / denominator
        return result

if __name__ == "__main__":
    test_parameters = [0.1, 0.01, 0.1, 0.01, 0]
    test_obj = CartPoleGame()
    next_params = test_obj.get_next_state_parameters(test_parameters, 10)
    print(f"Current params: {test_parameters}")
    print(f"Next params: {next_params}")
    for i in range(10):
        next_params = test_obj.get_next_state_parameters(next_params, 10)
        print(f"Loop params: {next_params}")