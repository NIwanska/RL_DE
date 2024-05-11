from differentialEvolution import DifferentialEvolution 
import numpy as np
import cec2017.functions as functions
import time

f = functions.f3

# Inicjalizacja obiektu klasy DifferentialEvolution
DE = DifferentialEvolution(
    objective_fun=f,
    popul_size=100,
    crossover_rate=0.5,
    max_iterations=20,
    bounds=(-100, 100),
    dimension=2,
    F=0.5,
    selection = 'best', #'rand',
    num_diff = 1  #2
)
start_time = time.time()
# Uruchomienie ewolucji
result, result_point, state = DE.evolve()

end_time = time.time()

# Obliczamy czas wykonania funkcji
execution_time = end_time - start_time

print("Czas wykonania funkcji:", execution_time, "sekund.")
print("Najlepszy wynik:", result)
print("Najlepsze rozwiązanie:", result_point)
print("Stan:", state)