from differentialEvolution import DifferentialEvolution
from env import Env
from qLearningSolver import QLearningSolver
from train import train_qsolver
import numpy as np
import matplotlib.pyplot as plt
import cec2017.functions as functions
from cec2017.utils import surface_plot
import time

f1 = functions.f3
f2 = functions.f8
f3 = functions.f5
f4 = functions.f15
f5 = functions.f25

# Inicjalizacja środowiska
env_handler = Env(f1, 100, 20, 10)

observation_space = env_handler.observation_space
q_params = [0.5, 0.9, 0.1]


# Inicjalizacja solvera
q_solver = QLearningSolver(observation_space,
                        q_params[0], q_params[1], q_params[2])


q_solver = train_qsolver(q_solver, env_handler, 4000, True, True)
print(q_solver.q_table)