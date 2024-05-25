
import matplotlib.pyplot as plt


class Plot_conv:
    def __init__(self):
        self.x = 0
        self.tests = 0
        self.x_max = 40
        self.tests_max = 100
        self.ys = [[] for _ in range(100)]
        self.fig, self.ax = plt.subplots()
        self.points, = self.ax.plot([], [], 'o-')
        self.x_data, self.y_data = [], []
        self.ax.set_xlim(0, self.x_max)
        self.legend = 'clear_de'
        self.min = float('inf')
        self.max= float('-inf')
        



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
        crossover_rate=0.5,
        max_iterations=k,
        bounds=(-100, 100),
        dimension=dim_f,
        F=0.8,
        selection = 'best', #'rand',
        num_diff = 1,  #2
        train = False

    )
    for _ in range(100):
        DE.initialize_popul()
        result, result_point, state = DE.evolve()

    with open(f'q_solver123_{dim_q}_{iter}.pkl', 'rb') as file:
        q_solver = pickle.load(file)

    for _ in range(100):
        env_handler = Env(func=f, population_size=100, iterrations_per_action=1, 
                        dimensions=dim_f, iterations_per_episode=k, train=False)

        q_solver_same = test_qsolver(q_solver, env_handler)

    plt.title(f'funkcja f{f} {dim_f} wymiarowa')

    plt.savefig(f'funkcja_{f}_{dim_f}_{iter}_{dim_q}.png')  

    plt.show()


for f in range(6):
    for iter in [1000, 5000]:
        for dim_q in [2, 10]:
            for dim_f    in [2, 10]:
                if dim_f == 2 and f == 3:
                    continue
                if dim_f == 2:
                    k = 40
                else:
                    k = 80
                testing(f, k, dim_f, dim_q, iter)
                plotter = Plot_conv()
