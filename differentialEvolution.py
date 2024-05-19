import numpy as np

class DifferentialEvolution:
    def __init__(
    self, 
    objective_fun,
    popul_size,
    crossover_rate,
    F,
    max_iterations,
    bounds,
    dimension,
    selection,
    num_diff
    ):
        self.objective_fun = objective_fun
        self.popul_size = popul_size
        self.crossover_rate = crossover_rate
        self.F = F
        self.max_iterations = max_iterations
        self.bounds = bounds
        self.num_params = dimension
        self.selection = selection
        self.num_diff = num_diff
        self.population = []
        self.obj_val =[]

    def initialize_popul(self):
        self.population = np.random.uniform(
            low=self.bounds[0], 
            high=self.bounds[1], 
            size=(self.popul_size, self.num_params))
        self.obj_val = self.objective_fun(self.population)
    
    def mutate(self, popul, best_from_popul):
        mutated_popul = np.copy(popul)
        for i in range(self.popul_size):
            if self.selection == 'best':
                a = best_from_popul
            elif self.selection == 'rand':
                 a = np.random.randint(0, self.popul_size)
            if self.num_diff == 1:
                b, c = np.random.choice(self.popul_size, 2, replace=False)
                mutated_popul[i] = popul[a] + self.F * (popul[b] - popul[c])
            elif self.num_diff == 2:    
                b, c, d, f = np.random.choice(self.popul_size, 4, replace=False)
                mutated_popul[i] = popul[a] + self.F * (popul[b] - popul[c]) + self.F * (popul[d] - popul[f])
            for coordinate in mutated_popul[i]:
                if coordinate < self.bounds[0]:
                    coordinate = self.bounds[0] + (self.bounds[0] - coordinate)
                elif coordinate > self.bounds[1]:
                    coordinate = self.bounds[1] - (coordinate - self.bounds[1])
        return mutated_popul
    


    def crossover(self, popul, mutated_popul):
        trial_popul = np.copy(popul)
        for i in range(self.popul_size):
            for j in range(self.num_params):
                if np.random.rand() < self.crossover_rate:
                    trial_popul[i, j] = mutated_popul[i, j]
        return trial_popul
    
    def avg_distance(self, popul):
        avg_distance = np.mean(np.linalg.norm(popul - popul.mean(axis=0), axis=1))
        return avg_distance
    

    def evolve(self):
        # self.population = self.initialize_popul()
        # self.obj_val = self.objective_fun(self.population)
        success_rate = 0
        for _ in range(self.max_iterations):
            mutated_popul = self.mutate(self.population, np.argmin(self.obj_val))
            trial_popul = self.crossover(self.population, mutated_popul)
            obj_val_trial = self.objective_fun(trial_popul)
            
            for i in range(self.popul_size):

                if obj_val_trial[i] < self.obj_val[i]:
                    success_rate += 1
                    self.population[i] = trial_popul[i]
                    self.obj_val[i] = obj_val_trial[i]

            print(min(self.obj_val))
            print(success_rate/((1+_)*self.popul_size))
        avg_distance = self.avg_distance(self.population)
        success_rate = success_rate/(self.max_iterations*self.popul_size)
            # Zwracanie procentowego sukcesu i średniej odległości jako stan
        state = (success_rate, avg_distance)  
        
        return min(self.obj_val), self.population[np.argmin(self.obj_val)], state
    
    def set_obj_function(self, func):
        self.objective_fun = func


