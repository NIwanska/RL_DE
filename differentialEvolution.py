import numpy as np

class DifferentialEvolution:
    def __init__(
    self, 
    objective_fun,
    pop_size,
    crossover_rate,
    F,
    max_iterations,
    bounds):
        self.objective_fun = objective_fun
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.F = F
        self.max_iterations = max_iterations
        self.bounds = bounds
        self.num_params = len(bounds)
        
    def initialize_pop(self):
        return np.random.uniform(
            low=self.bounds[0], 
            high=self.bounds[1], 
            size=(self.pop_size, self.num_params))
    
    def mutate(self, pop):
        mutated_pop = np.copy(pop)
        for i in range(self.pop_size):
            a, b, c = np.random.choice(self.pop_size, 3, replace=False)
            mutated_pop[i] = pop[a] + self.F * (pop[b] - pop[c])
        return mutated_pop
    
    def crossover(self, pop, mutated_pop):
        trial_pop = np.copy(pop)
        for i in range(self.pop_size):
            for j in range(self.num_params):
                if np.random.rand() < self.crossover_rate:
                    trial_pop[i, j] = mutated_pop[i, j]
        return trial_pop
    
    def evolve(self):
        pop = self.initialize_pop()
        for _ in range(self.max_iterations):
            mutated_pop = self.mutate(pop)
            trial_pop = self.crossover(pop, mutated_pop)
            for i in range(self.pop_size):
                obj_val_current = self.objective_fun(pop[i])
                obj_val_trial = self.objective_fun(trial_pop[i])
                if obj_val_trial < obj_val_current:
                    pop[i] = trial_pop[i]
        return pop[np.argmin([self.objective_fun(i) for i in pop])]


