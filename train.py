from env import Env
from qLearningSolver import QLearningSolver
import random as rd
import matplotlib.pyplot as plt

def train_qsolver(q_solver: QLearningSolver,env_handler: Env, q_params, learning_iter, plots: bool, verbose:
                                                        bool=False):
    # observation_space = env_handler.observation_space

    # Trening
    num_episodes = learning_iter
    all_rewards = []
    for episode in range(num_episodes):
        state = env_handler.reset()
        done = False
        total_reward = 0
        while not done:
            # Wybór akcji
            if q_solver.epsilon > rd.random():
                action = env_handler.action_sample()
            else:
                action = q_solver.get_best_action(state)
            
            # Wykonanie akcji
            next_state, reward, done= env_handler.step(action)[0:3]

            # Dyskretyzacja
            closest_key_suc_rate = min(q_solver.keys(), key=lambda k: abs(k[0] - next_state[0]))
            closest_key_avg_dist = min(q_solver.keys(), key=lambda k: abs(k[1] - next_state[1]))
            next_state = (closest_key_suc_rate, closest_key_avg_dist) 

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