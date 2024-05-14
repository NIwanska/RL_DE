from differentialEvolution import DifferentialEvolution
import numpy as np
import random

class Env:

    def __init__(self, func, population_size, iterrations_per_action, dimensions) -> None:
        self.func = func
        self.de = DifferentialEvolution(
            objective_fun=func,
            popul_size=population_size,
            crossover_rate=0.5,
            max_iterations=iterrations_per_action,
            bounds=(-100, 100),
            dimension=dimensions,
            F=0.8,
            selection = 'best', #'rand',
            num_diff = 1  #2
        )
        self.safed_de_state = None
        self.observation_space = self.create_obs_space(dimensions)
        # self.de.initialize_popul()
        self.actions_counter = 0
        self.prev_result = None

    def action(self, action_idx):
        match action_idx:
            case 0:
                self.de.F += 0.2
            
            case 1:
                self.de.F -= 0.2

            case 2:
                self.de.selection = 'best'

            case 3:
                self.de.selection = 'rand'

            case 4:
                self.de.num_diff = 1

            case 5:
                self.de.num_diff = 2

    def safe_de_state(self):
        self.safe_de_state = self.de.population

    def create_obs_space(self, dimensions):
        values1 = np.arange(0, 1.05, 0.05)  # od 0 do 1 z rozdzielczością 0.05
        values2 = np.linspace(0, 200*np.sqrt(dimensions), 20)  # od 0 do dim, 20 wartości
        values3 = np.arange(0, 6)  # od 0 do 5, całkowite
        return (values1, values2, values3)
        
    def reset(self):
        self.de.initialize_popul()

    def action_sample(self):
        return random.randint(0, 5)

    def step(self, action):
        done = False
        self.action(action)
        result, _, next_state = self.de.evolve()
        reward = next_state[0] * 10 # - result

        if self.actions_counter > 200:
            done = True
        
        if self.prev_result is not None:
            if result - self.prev_result:
                done = True
        self.prev_result = result

        return (next_state, reward, done)
    
    def get_first_state(self):
        avg_distance = self.de.avg_distance(self.de.population)
        return (0, avg_distance)
    
    