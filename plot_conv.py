import matplotlib.pyplot as plt

class Plot_conv:
    def __init__(self):
        self.x = 0
        self.tests = 0
        self.ys = []
        self.fig, self.ax = plt.subplots()
        self.points, = self.ax.plot([], [], 'o-')
        self.x_data, self.y_data = [], []
        self.ax.set_xlim(0, 20)
        self.legend = 'clear_de'
        self.min = float('inf')
        self.max= float('-inf')
        


plotter = Plot_conv()

def add_point(y):

    if plotter.tests < 20:
        plotter.ys.append(y)
    else:
        plotter.ys[plotter.x] += y

    plotter.x += 1

    if plotter.x == 20:
        plotter.x = 0
        # print('x=0')
    plotter.tests +=1
    if plotter.tests == 20*100:
        plotter.ys = [x / 100 for x in plotter.ys]
        print(plotter.ys[-1])
        plotter.min = min(plotter.min,min(plotter.ys))
        plotter.max = max(plotter.max,max(plotter.ys))
        plotter.ax.set_ylim(plotter.min, plotter.max)
        plotter.tests = 0
        plotter.ax.plot(range(20), plotter.ys, 'o-', label=plotter.legend)
        plotter.ax.legend()
        plotter.fig.canvas.draw()
        plotter.ys = []
        plotter.legend = 'qlearning_de'

    
