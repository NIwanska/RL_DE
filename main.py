from differentialEvolution import DifferentialEvolution 
import cec2017.functions as functions
from train import train_qsolver, test_qsolver
import pickle
from env import Env
import matplotlib.pyplot as plt

f1 = functions.f3

# Inicjalizacja obiektu klasy DifferentialEvolution
DE = DifferentialEvolution(
    objective_fun=f1,
    popul_size=100,
    crossover_rate=0.5,
    max_iterations=20,
    bounds=(-100, 100),
    dimension=10,
    F=0.5,
    selection = 'best', #'rand',
    num_diff = 1,  #2
    train = False

)
for _ in range(100):
    DE.initialize_popul()

    # Uruchomienie ewolucji
    result, result_point, state = DE.evolve()



with open('q_solver.pkl', 'rb') as file:
    q_solver = pickle.load(file)

for _ in range(100):
    env_handler = Env(func=f1, population_size=100, iterrations_per_action=1, 
                    dimensions=10, iterations_per_episode=20, train=False)

    test_qsolver(q_solver, env_handler)

plt.show()