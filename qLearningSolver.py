import random as rd
from env import Env
import matplotlib.pyplot as plt

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





def train_qsolver(q_solver: QLearningSolver,env_handler: Env, learning_iter, plots: bool, verbose:
                                                        bool=False):
    # observation_space = env_handler.observation_space

    # Trening
    num_episodes = learning_iter
    all_rewards = []
    for episode in range(num_episodes):
        env_handler.reset()
        state = env_handler.get_first_state()
        closest_key_avg_dist = min(q_solver.keys(), key=lambda k: abs(k[1] - state[1]))
        state = (0, closest_key_avg_dist[1])
        done = False
        total_reward = 0
        while not done:
            # Wybór akcji
            if q_solver.epsilon > rd.random():
                action = env_handler.action_sample()
            else:
                action = q_solver.get_best_action(state)
            
            # Wykonanie akcji
            next_state, reward, done= env_handler.step(action)

            # Dyskretyzacja
            closest_key_suc_rate = min(q_solver.keys(), key=lambda k: abs(k[0] - next_state[0]))
            closest_key_avg_dist = min(q_solver.keys(), key=lambda k: abs(k[1] - next_state[1]))
            next_state = (closest_key_suc_rate[0], closest_key_avg_dist[1]) 

            # Update wartości q
            q_solver.update(state, action, reward, next_state)

            total_reward += reward
            state = next_state    

        all_rewards.append(total_reward)
        if episode % 10 == 0 and verbose:
            print(f"Episode: {episode}, Total Reward: {total_reward}")
    if plots:
        fig1 = plt.plot(all_rewards)
        plt.xlabel('Epizod')
        plt.ylabel('suma nagród')
        plt.title(f'Wykres sumy nagród uzyskanych w poszczególnych iteracjach uczących\nLiczba iteracji uczących: {num_episodes}')
        plt.show()
    return q_solver


def test_qsolver(q_solver: QLearningSolver,env_handler: Env):
    env_handler.reset()
    state = env_handler.get_first_state()
    closest_key_avg_dist = min(q_solver.keys(), key=lambda k: abs(k[1] - state[1]))
    state = (0, closest_key_avg_dist[1])
    done = False
    while not done:
        action = q_solver.get_best_action(state)
        
        # Wykonanie akcji
        next_state, reward, done= env_handler.step(action)

        # Dyskretyzacja
        closest_key_suc_rate = min(q_solver.keys(), key=lambda k: abs(k[0] - next_state[0]))
        closest_key_avg_dist = min(q_solver.keys(), key=lambda k: abs(k[1] - next_state[1]))
        next_state = (closest_key_suc_rate[0], closest_key_avg_dist[1]) 

        state = next_state    

   
    return q_solver