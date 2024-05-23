from scipy.optimize import differential_evolution
import cec2017.functions as functions
import numpy as np
import matplotlib.pyplot as plt

# Wybór funkcji celu
f1 = functions.f3
f2 = functions.f8
f3 = functions.f5
f4 = functions.f15
f5 = functions.f25
f6 = functions.f10

func = f2

# Zakresy dla poszczególnych zmiennych
bounds = [(-100, 100)] * 2
progress = []

# Wrapper dla funkcji celu, aby przekształcić x na odpowiedni kształt
def func_wrapper(x):
    x = np.reshape(x, (1, -1))  # Przekształcenie do kształtu (1, n)
    return func(x)

# Funkcja zwrotna wywoływana po każdej iteracji
def callback(xk, convergence):
    best_value = func_wrapper(xk)
    progress.append(best_value)
    print(f"Iteration {len(progress)}: {best_value}")

# Wywołanie algorytmu różnicowej ewolucji z funkcją zwrotną
result = differential_evolution(func_wrapper, bounds,callback=callback, maxiter=20)

# Wyświetlenie wyników
print("Optymalne rozwiązanie:", result.x)
print("Wartość funkcji celu:", result.fun)

# Rysowanie wykresu postępu
plt.plot(progress, marker='o')
plt.xlabel('Iteracja')
plt.ylabel('Wartość funkcji celu')
plt.title('Postęp algorytmu różnicowej ewolucji')
plt.grid(True)
plt.show()
