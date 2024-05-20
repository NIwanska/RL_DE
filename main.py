from differentialEvolution import DifferentialEvolution 
import cec2017.functions as functions
from qLearningSolver import test_qsolver
import pickle
from env import Env
import matplotlib.pyplot as plt

f1 = functions.f3
f2 = functions.f8
f3 = functions.f5
f4 = functions.f15
f5 = functions.f25
f6 = functions.f10

f = f1
dimensions=10


# Inicjalizacja obiektu klasy DifferentialEvolution
DE = DifferentialEvolution(
    objective_fun=f,
    popul_size=100,
    crossover_rate=0.5,
    max_iterations=80,
    bounds=(-100, 100),
    dimension=dimensions,
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
    env_handler = Env(func=f, population_size=100, iterrations_per_action=1, 
                    dimensions=dimensions, iterations_per_episode=80, train=False)

    q_solver_same = test_qsolver(q_solver, env_handler)

# for x in (q_solver_same.q_table):
#     if(q_solver_same.q_table[x]) != 0:
#         # print(x)
#         # print(q_solver_same.q_table[x])
# print(q_solver_same.q_table)
plt.title(f'funkcja f0')

# Zapisywanie wykresu do pliku (np. PNG, PDF, SVG)
plt.savefig(f'funkcja_f0.png')  # Można zmienić rozszerzenie na .pdf, .svg itp.

plt.show()