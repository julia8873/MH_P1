import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Ajustar los nombres para que coincidan con el C++ (nombres = { "Random", "Greedy", "BL" })
algoritmos = ["Random", "Greedy", "BL", "BL_NoOpt"]
data_dict = {}

# 2. Carga de datos
for algo in algoritmos:
    filename = f"fitness_{algo}.csv"
    if os.path.exists(filename):
        # Leemos el fitness (uno por línea) [cite: 875]
        data_dict[algo] = pd.read_csv(filename, header=None)[0]
    else:
        print(f"Aviso: No se encontró el archivo {filename}")

# 3. Creación del DataFrame y Gráfica
if data_dict:
    df = pd.DataFrame(data_dict)

    plt.figure(figsize=(10, 6))
    
    # Solo graficamos las columnas que realmente se cargaron
    columnas_reales = [a for a in algoritmos if a in df.columns]
    
    df.boxplot(column=columnas_reales, grid=False, patch_artist=True, 
               boxprops=dict(facecolor="#3478a8", color="black"),
               medianprops=dict(color="black"))

    plt.title('Distribución de Fitness: Zoo (15% restricciones)')
    plt.xlabel('Algoritmo')
    plt.ylabel('Fitness')

    plt.savefig('boxplot_fitness_zoo.png', dpi=300)
    print("Gráfica generada exitosamente como 'boxplot_fitness_zoo.png'")
    plt.show()
else:
    print("Error: No se encontró ningún archivo .csv en esta carpeta.")