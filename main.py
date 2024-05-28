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

fs = [f1, f2, f3, f4, f5, f6]

def testing(f, k, dim_f, dim_q, iter):
    DE = DifferentialEvolution(
        objective_fun=fs[f],
        popul_size=100,
        crossover_rate=0.8,
        max_iterations=k,
        bounds=(-100, 100),
        dimension=dim_f,
        F=0.8,
        selection = 'best', #'rand',
        num_diff = 1,  #2
        train = False

    )
    for _ in range(50):
        DE.initialize_popul()
        DE.evolve()

    with open(f'q_solver123_{dim_q}_{iter}.pkl', 'rb') as file:
        q_solver = pickle.load(file)

    for _ in range(50):
        env_handler = Env(func=fs[f], population_size=100, iterrations_per_action=1, 
                        dimensions=dim_f, iterations_per_episode=k, train=False)

        test_qsolver(q_solver, env_handler)

    plt.title(f'Krzywa ECDF: funkcja f{f+1} {dim_f} wymiarowa')
    
    plt.savefig(f'funkcja_{f+1}_{dim_f}_{iter}_{dim_q}.png')
    # plt.show() 


for f in range(6):
    for iter in [1000, 5000]:
        for dim_q in [2, 10]:
            for dim_f    in [10]:
                if dim_f == 2:
                    k = 40
                else:
                    k = 100
                testing(f, k, dim_f, dim_q, iter)