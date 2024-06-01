import matplotlib.pyplot as plt
import numpy as np
import cec2017.functions as functions

f1 = functions.f3
f2 = functions.f8
f3 = functions.f5
f4 = functions.f15
f5 = functions.f25
f6 = functions.f10



class Plot_conv:
    def __init__(self):
        self.tests_2 =0
        self.x = 0
        self.tests = 0
        self.x_max = 1000
        self.tests_max = 50
        self.ys = [[] for _ in range(self.tests_max)]
        self.fig, self.ax = plt.subplots()
        self.x_data, self.y_data = [], []
        # self.ax.set_xlim(0, self.x_max)
        self.legend = 'clear_de'
        self.min = float('inf')
        self.max= float('-inf')
        self.ax.legend(loc='lower right')
        self.ax.set_xlabel('Data values')
        self.ax.set_ylabel('ECDF')
        


plotter = Plot_conv()

def add_point(y, f):
    if plotter.tests_2 == 2*plotter.x_max*plotter.tests_max:
        plotter.fig, plotter.ax = plt.subplots()
        # plotter.ax.set_xlim(0, plotter.x_max)
        plotter.legend = 'clear_de'
        plotter.min = float('inf')
        plotter.max= float('-inf')
        plotter.tests_2 = 0
    plotter.ys[plotter.tests // plotter.x_max].append(y)

    plotter.tests +=1
    plotter.tests_2 +=1
    if plotter.tests == plotter.x_max*plotter.tests_max:
        Y = np.array(plotter.ys)

        means = np.mean(Y, axis=0)
        conv = np.std(Y, axis=0)
        mins = np.min(Y, axis=0)
        maxs = np.max(Y, axis=0)
        x = np.sort(means)[10:]

        y = np.arange(1, len(x) + 1) / len(x)
        print(means[-1])
        zapisz_dane_do_pliku(means[-1], conv, mins, maxs, plotter.legend, f)
        # plotter.min = min(plotter.min, min(y))
        # plotter.max = max(plotter.max, max(y))
        # plotter.ax.set_ylim(plotter.min, plotter.max)
        plotter.ax.plot(x, y, marker='.', linestyle='none', label=plotter.legend)
        # plotter.min = min(plotter.min,min(means))
        # plotter.max = max(plotter.max,max(means))
        # plotter.ax.set_ylim(plotter.min, plotter.max)
        # plotter.ax.plot(range(plotter.x_max), means, 'o-', label=plotter.legend)

        plotter.tests = 0
        plotter.ax.legend()
        plotter.ys = [[] for _ in range(plotter.tests_max)]
        plotter.legend = 'qlearning_de'
        
    
def zapisz_dane_do_pliku(mean, conv, mins, maxs, type, f):
    with open('wyniki.txt', 'a') as file:
        if f == f1:
            file.write(f'\n\nf1\n')
        elif f == f2:   
            file.write(f'\n\nf2\n')
        elif f == f3:
            file.write(f'\n\nf3\n')
        elif f == f4:
            file.write(f'\n\nf4\n')
        elif f == f5:
            file.write(f'\n\nf5\n')
        elif f == f6:
            file.write(f'\n\nf6\n')

        file.write(type)
        file.write(f'\nlast_mean')
        file.write(f'\n{mean}')
        file.write(f'\nconv') 
        file.write(f'\n{conv[-1]}')
        file.write(f'\nmins')
        file.write(f'\n{mins[-1]}')
        file.write(f'\nmaxs')
        file.write(f'\n{maxs[-1]}')
        file.write(f'\n\n') 