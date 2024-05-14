import numpy as np
import random as rd
from env import Env

class QLearningSolver:

    def __init__(
        self,
        observation_space: tuple,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {(v1, v2, v3): 0 for v1 in observation_space[0] 
                        for v2 in observation_space[1] for v3 in observation_space[2]}

    def __call__(self, state: tuple, action: int) -> float:
        return self.q_table[(state[0], state[1], action)]
    
    def keys(self):
        return self.q_table.keys()

    def update(self, state: tuple, action: int, reward:
                                    float, next_state: tuple) -> None:
        current_q: float = self.q_table[(state[0], state[1], action)]
        temp_next = [self.q_table[(next_state[0], next_state[1], i)] for i in range(6)]
        max_next_q: float = max(temp_next)
        new_q: float = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state[0], state[1], action)] = new_q

    def get_best_action(self, state: tuple) -> int:
        filtered_keys = [k for k in self.q_table if k[0] == state[0] and k[1] == state[1]]

        return max(filtered_keys, key=self.q_table.get)[2]





