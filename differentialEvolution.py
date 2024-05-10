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
    dimension):
        self.objective_fun = objective_fun
        self.popul_size = popul_size
        self.crossover_rate = crossover_rate
        self.F = F
        self.max_iterations = max_iterations
        self.bounds = bounds
        self.num_params = dimension
        
    def initialize_popul(self):
        return np.random.uniform(
            low=self.bounds[0], 
            high=self.bounds[1], 
            size=(self.popul_size, self.num_params))
    
    def mutate(self, popul):
        mutated_popul = np.copy(popul)
        for i in range(self.popul_size):
            a, b, c = np.random.choice(self.popul_size, 3, replace=False)
            mutated_popul[i] = popul[a] + self.F * (popul[b] - popul[c])
        return mutated_popul
    
    def crossover(self, popul, mutated_popul):
        trial_popul = np.copy(popul)
        for i in range(self.popul_size):
            for j in range(self.num_params):
                if np.random.rand() < self.crossover_rate:
                    trial_popul[i, j] = mutated_popul[i, j]
        return trial_popul
    
    def evolve(self):
        popul = self.initialize_popul()
        obj_val = self.objective_fun(popul)
        for _ in range(self.max_iterations):
            mutated_popul = self.mutate(popul)
            trial_popul = self.crossover(popul, mutated_popul)
            obj_val_trial = self.objective_fun(trial_popul)
            for i in range(self.popul_size):               
                if obj_val_trial[i] < obj_val[i]:
                    popul[i] = trial_popul[i]
                    obj_val[i] = obj_val_trial[i]
        return min(obj_val), popul[np.argmin(obj_val)]


