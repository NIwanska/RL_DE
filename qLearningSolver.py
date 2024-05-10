import numpy as np
import random as rd

class QLearningSolver:

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((observation_space, action_space))

    def __call__(self, state: np.ndarray, action: np.ndarray) ->                                                            np.ndarray:
        return self.q_table[state, action]

    def update(self, state: np.ndarray, action: np.ndarray, reward:
                                    float, next_state: int) -> None:
        current_q: np.ndarray = self.q_table[state, action]
        max_next_q: float = np.max(self.q_table[next_state])
        new_q: np.ndarray = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

    def get_best_action(self, state: np.ndarray) -> np.ndarray:      
        return np.argmax(self.q_table[state])


def train(env_handler, q_params, learning_iter, plots: bool, verbose:
                                                        bool=False):
    observation_space = env_handler.observation_space.n
    action_space = env_handler.action_space.n

    # Inicjalizacja solvera
    q_solver = QLearningSolver(observation_space, action_space,
                            q_params[0], q_params[1], q_params[2])
    # Trening
    num_episodes = learning_iter
    for _ in range(num_episodes):
        state = env_handler.reset()[0]
        done = False

        while not done:
            # Wybór akcji
            if q_solver.epsilon > rd.random():
                action = env_handler.action_space.sample()
            else:
                action = q_solver.get_best_action(state)
            
            # Wykonanie akcji
            next_state, reward, done= env_handler.step(action)[0:3]
            
            # Update wartości q
            q_solver.update(state, action, reward, next_state)
            state = next_state        


