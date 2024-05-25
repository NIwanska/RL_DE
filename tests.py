from env import Env
from qLearningSolver import QLearningSolver, train_qsolver
import cec2017.functions as functions
from cec2017.utils import surface_plot
import pickle

f1 = functions.f3
f2 = functions.f8
f3 = functions.f5
f4 = functions.f15
f5 = functions.f25

fs = [f1, f2, f3]


env_handler = Env(func=f1, population_size=100, iterrations_per_action=1, 
                  dimensions=10, iterations_per_episode=80, train=True)

observation_space = env_handler.observation_space

# lr, gamma, epsilon
q_params = [0.5, 0.9, 0.5]


q_solver = QLearningSolver(observation_space,
                        q_params[0], q_params[1], q_params[2])


q_solver = train_qsolver(q_solver, env_handler, 5000, fs, False, True)



with open('q_solver123_10_5000.pkl', 'wb') as file:
    pickle.dump(q_solver, file)


not_zero = 0
all = 0
for k in q_solver.q_table:
    if q_solver.q_table[k] != 0:
        not_zero += 1
    all += 1

print(f'Ilość z wart q: {not_zero}; ilość wszystkich: {all}')


