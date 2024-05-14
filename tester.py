from differentialEvolution import DifferentialEvolution
from env import Env
from qLearningSolver import QLearningSolver
from train import train_qsolver
import numpy as np
import matplotlib.pyplot as plt
import cec2017.functions as functions
from cec2017.utils import surface_plot
import time

def tester(q_solver:QLearningSolver, env_handler: Env, testing_iter: int, plots: bool, 
           verbose: bool=False):
    test_rewards = np.array([])
    for episode in range(testing_iter):
        state = env_handler.reset()
        done = False
        total_reward = 0
        prev_state = None
        prev_state2 = False
        
        while not done:
            if prev_state == state or prev_state2 == prev_state:
                action = env_handler.action_space.sample()
            else:
                action = q_solver.get_best_action(state)
            # print(f'stan:{state}; akcja: {action}')
            next_state, reward, done= env_handler.step(action)
            total_reward += reward
            prev_state2 = prev_state
            prev_state = state
            state = next_state

        test_rewards = np.append(test_rewards, [total_reward])
        if verbose:
            print(f"Test Episode: {episode}, Total Reward: {total_reward}")
            
    print(f'Średnia wartość nagrody: {np.mean(test_rewards)}')
    if plots:
        fig2 = plt.plot(test_rewards)
        plt.ylim([-10, 20])
        plt.xlabel('Epizod')
        plt.ylabel('Suma nagród')
        plt.title(f'Wykres sumy nagród uzyskanych przy rozwiązywaniu problemu\nLiczba podejść = {testing_iter}')
        plt.show()