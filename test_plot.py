from differentialEvolution import DifferentialEvolution 
import cec2017.functions as functions
import matplotlib.pyplot as plt
import numpy as np


f = functions.f25 # f8

def contour_plot_1(function, domain=(-100,100), points=30, ax=None, dimension=2, population=None):
    xys = np.linspace(domain[0], domain[1], points)
    xys = np.transpose([np.tile(xys, len(xys)), np.repeat(xys, len(xys))])

    if dimension > 2:
        tail = np.zeros((xys.shape[0], dimension - 2))
        x = np.concatenate([xys, tail], axis=1)
        zs = function(x)
    else:
        zs = function(xys)

    fig = plt.figure()
    ax = fig.gca()

    X = xys[:,0].reshape((points, points))
    Y = xys[:,1].reshape((points, points))
    Z = zs.reshape((points, points))
    cont = ax.contourf(X, Y, Z, lw = 1, levels=20, cmap='plasma')
    ax.contour(X, Y, Z, colors="k", linestyles="solid")
    cbar = fig.colorbar(cont, shrink=0.5, aspect=5, pad=0.15, label='')

    ax.set_title(function.__name__)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    x1 = []
    x2 = []
    for i in range(len(pop)):
        temp1 = []
        temp2 = []
        for j in range(len(pop[i])):
            val1 = max(-100, min(pop[i][j][0], 100))
            val2 = max(-100, min(pop[i][j][1], 100))
            temp1.append(val1)
            temp2.append(val2)
        x1.append(temp1)
        x2.append(temp2)
    for i in range(5):
        scatter = ax.scatter(x1[i], x2[i],color='red', zorder=1)
        plt.title(f'Generation: {i*10}')
        plt.savefig(f'C:/Users/fszcz/uma/test_plots/fig{i}.png')
        plt.pause(0.001)
        scatter.remove()

    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    plt.show()
    
    return fig



DE = DifferentialEvolution(
        objective_fun=f,
        popul_size=25,
        crossover_rate=0.8,
        max_iterations=1,
        bounds=(-100, 100),
        dimension=2,
        F=0.8,
        selection = 'best', #'rand',
        num_diff = 1,  #2
        train = False

    )

pop = []
DE.initialize_popul()
for i in range(51):
    DE.evolve()
    if i % 10 == 0:
        pop.append(DE.population.copy())


contour_plot_1(f, points=120, population=pop)

