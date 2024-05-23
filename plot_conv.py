import matplotlib.pyplot as plt
import numpy as np

class Plot_conv:
    def __init__(self):
        self.tests_2 =0
        self.x = 0
        self.tests = 0
        self.x_max = 100
        self.tests_max = 100
        self.ys = [[] for _ in range(100)]
        self.fig, self.ax = plt.subplots()
        # self.points, = self.ax.plot([], [], 'o-')
        self.x_data, self.y_data = [], []
        self.ax.set_xlim(0, self.x_max)
        self.legend = 'clear_de'
        self.min = float('inf')
        self.max= float('-inf')
        


plotter = Plot_conv()

def add_point(y):
    if plotter.tests_2 == 2*plotter.x_max*plotter.tests_max:
        # Stwórz nowe okno wykresu
        # plotter.ys = [[] for _ in range(100)]
        plotter.fig, plotter.ax = plt.subplots()
        # plotter.points, = plotter.ax.plot([], [], 'o-')
        # plotter.x_data, plotter.y_data = [], []
        plotter.ax.set_xlim(0, plotter.x_max)
        plotter.legend = 'clear_de'
        plotter.min = float('inf')
        plotter.max= float('-inf')
        plotter.tests_2 = 0
    # print(y)
    plotter.ys[plotter.tests // plotter.x_max].append(y)

    plotter.tests +=1
    plotter.tests_2 +=1
    if plotter.tests == plotter.x_max*plotter.tests_max:
        Y = np.array(plotter.ys)

        # Obliczenie średnich dla każdej z 20 kolumn
        means = np.mean(Y, axis=0)
        conv = np.std(Y, axis=0)
        mins = np.min(Y, axis=0)
        maxs = np.max(Y, axis=0)
        print(means[-1])
        zapisz_dane_do_pliku(means, conv, mins, maxs, plotter.legend)
        plotter.min = min(plotter.min,min(means))
        plotter.max = max(plotter.max,max(means))
        plotter.ax.set_ylim(plotter.min, plotter.max)
        plotter.tests = 0
        plotter.ax.plot(range(plotter.x_max), means, 'o-', label=plotter.legend)
        plotter.ax.legend()
        # plotter.fig.canvas.draw()
        plotter.ys = [[] for _ in range(100)]
        plotter.legend = 'qlearning_de'
        
    
def zapisz_dane_do_pliku(means, conv, mins, maxs, type):
    with open('wyniki.txt', 'a') as file:
        file.write(type)
        for y in means:
            file.write(f'\n{y}')
        file.write(f'\nconv') 
        file.write(f'\n{conv[-1]}')
        file.write(f'\nmins')
        file.write(f'\n{mins[-1]}')
        file.write(f'\nmaxs')
        file.write(f'\n{maxs[-1]}')
        file.write(f'\n\n') 