import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
from sklearn.decomposition import PCA

# Configuración de estilo
sns.set_theme(style="whitegrid")

# Parámetros globales
algoritmos = ["Random", "Greedy", "BL", "BL_NoOpt"]
colores_boxplot = ["#3478a8", "#e38633", "#32a852", "#a83232"] 

# Ajusta el valor de K para tus datasets aquí
K_VALUES = {
    "zoo": 7, 
    "glass": 7, 
    "bupa": 16, 
    "test": 2  # Cambiado a 2 según tu ejemplo de dos grupos
}

def plot_all_results():
    # 1. Detectar carpetas de resultados
    folders = glob.glob("results_*")
    
    if not folders:
        print("No se detectaron carpetas que empiecen por 'results_'.")
        print(f"Directorio actual: {os.getcwd()}")
        return

    for folder in folders:
        print(f"\n--- Procesando: {folder} ---")
        
        # Extraer metadatos del nombre de la carpeta
        parts = folder.split("_")
        if len(parts) < 2: continue
        
        dataset_name = parts[1].lower()
        tag = parts[-1]
        k = K_VALUES.get(dataset_name, 7)

        # Definir rutas de archivos de datos (Ajusta estas rutas si es necesario)
        data_path = f"../data/{dataset_name}_set.dat"
        const_path = f"../data/{dataset_name}_set_const_{tag}.dat"

        # --- SECCIÓN 1: BOXPLOTS (FITNESS Y TIEMPOS) ---
        data_fit = {}
        data_time = {}
        for algo in algoritmos:
            f_fit = os.path.join(folder, f"fitness_{algo}.csv")
            f_time = os.path.join(folder, f"times_{algo}.csv")
            if os.path.exists(f_fit): 
                data_fit[algo] = pd.read_csv(f_fit, header=None)[0]
            if os.path.exists(f_time): 
                data_time[algo] = pd.read_csv(f_time, header=None)[0]

        if data_fit:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=pd.DataFrame(data_fit), palette=colores_boxplot)
            plt.title(f'Distribución de Fitness - {dataset_name.upper()}')
            plt.ylabel('Fitness (Menor es mejor)')
            plt.savefig(os.path.join(folder, "boxplot_fitness.png"))
            plt.close()

        if data_time:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=pd.DataFrame(data_time), palette=colores_boxplot)
            plt.yscale('log')
            plt.title(f'Tiempos de Ejecución - {dataset_name.upper()}')
            plt.ylabel('Segundos (Escala Log)')
            plt.savefig(os.path.join(folder, "boxplot_tiempos.png"))
            plt.close()

        # --- SECCIÓN 2: PCA Y VISUALIZACIÓN DE CLUSTERS ---
        if os.path.exists(data_path):
            # Cargar datos de puntos
            X = pd.read_csv(data_path, header=None, sep=None, engine='python')
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)

            # Cargar restricciones de forma robusta (Solo las 3 primeras columnas)
            constraints = []
            if os.path.exists(const_path):
                try:
                    df_const = pd.read_csv(const_path, header=None, sep=None, engine='python')
                    constraints = df_const.iloc[:, :3].values
                    print(f"-> {len(constraints)} restricciones cargadas.")
                except Exception as e:
                    print(f"-> Error al leer restricciones: {e}")

            # Configurar cuadrícula de subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            fig.suptitle(f'Clusters, Etiquetas y Restricciones: {dataset_name.upper()} (k={k})', fontsize=18)

            for i, algo in enumerate(algoritmos):
                sol_path = os.path.join(folder, f"best_sol_{algo}.csv")
                
                # Dibujar puntos
                if os.path.exists(sol_path):
                    labels = pd.read_csv(sol_path, header=None)[0].values
                    scatter = axes[i].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, 
                                            cmap='tab10', s=180, edgecolors='black', zorder=5)
                else:
                    axes[i].scatter(X_2d[:, 0], X_2d[:, 1], c='gray', alpha=0.3, s=180, zorder=5)

                # Añadir etiquetas de texto x0, x1, ...
                for idx in range(len(X_2d)):
                    axes[i].annotate(f'x{idx}', (X_2d[idx, 0], X_2d[idx, 1]), 
                                     textcoords="offset points", xytext=(0,12), 
                                     ha='center', fontsize=10, fontweight='bold', zorder=6)

                # Dibujar líneas de restricciones
                for row in constraints:
                    if len(row) < 3: continue
                    p1, p2, c_type = int(row[0]), int(row[1]), row[2]
                    
                    if p1 < len(X_2d) and p2 < len(X_2d):
                        x_coords = [X_2d[p1, 0], X_2d[p2, 0]]
                        y_coords = [X_2d[p1, 1], X_2d[p2, 1]]
                        
                        if c_type == 1: # Must-Link
                            axes[i].plot(x_coords, y_coords, color='green', 
                                         linestyle='-', linewidth=2, alpha=0.7, zorder=2)
                        elif c_type == -1: # Cannot-Link
                            axes[i].plot(x_coords, y_coords, color='red', 
                                         linestyle='--', linewidth=2, alpha=0.7, zorder=2)

                axes[i].set_title(f"Algoritmo: {algo}", fontsize=14)
                axes[i].set_xlabel("Componente Principal 1")
                axes[i].set_ylabel("Componente Principal 2")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(folder, "cluster_comparison_full.png"))
            plt.close()
            print(f"-> Gráfica de clusters generada en {folder}")
        else:
            print(f"-> Error: No se encontró el dataset en {data_path}")

if __name__ == "__main__":
    plot_all_results()