import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import numpy as np
import os
import glob

# =====================================================================
# CONSTANTES DE FILTRADO
# =====================================================================
# Algoritmos que aparecen en TODAS las gráficas
ALGOS_TODAS = ['GREEDY', 'RANDOM', 'BL', 'EXTRA']
# Algoritmos que aparecen SOLO en gráficas de Tiempo
ALGOS_SOLO_TIEMPO = ['BL_OPTIMIZADO']
# Orden visual en las gráficas
ORDEN_ALGOS = ['GREEDY', 'RANDOM', 'BL', 'BL_OPTIMIZADO', 'EXTRA']

def filtrar_algos(df, col_algo, solo_tiempo=False):
    """
    solo_tiempo=False → incluye ALGOS_TODAS (sin BL_OPTIMIZADO)
    solo_tiempo=True  → incluye ALGOS_TODAS + ALGOS_SOLO_TIEMPO
    """
    if solo_tiempo:
        permitidos = ALGOS_TODAS + ALGOS_SOLO_TIEMPO
    else:
        permitidos = ALGOS_TODAS
    return df[df[col_algo].isin(permitidos)]

def orden_presente(df, col_algo):
    """Devuelve ORDEN_ALGOS filtrado a los que realmente están en el df."""
    presentes = df[col_algo].unique()
    return [a for a in ORDEN_ALGOS if a in presentes]

# =====================================================================
# 1. FUNCIÓN DE RE-ETIQUETADO
# =====================================================================
def match_clusters_to_reference(df_sol, df_ref_sol):
    c_ref = df_ref_sol['Cluster'].values
    c_sol = df_sol['Cluster'].values
    K = len(np.unique(c_ref))
    if len(c_ref) != len(c_sol):
        return c_sol

    labels_ref = np.unique(c_ref)
    labels_sol = np.unique(c_sol)
    label_to_matrix_idx_ref = {label: i for i, label in enumerate(labels_ref)}
    label_to_matrix_idx_sol = {label: i for i, label in enumerate(labels_sol)}
    matrix_idx_to_label_ref = {i: label for i, label in enumerate(labels_ref)}
    matrix_idx_to_label_sol = {i: label for i, label in enumerate(labels_sol)}

    intersection_matrix = np.zeros((K, K), dtype=int)
    for i in range(len(c_ref)):
        idx_ref = label_to_matrix_idx_ref[c_ref[i]]
        idx_sol = label_to_matrix_idx_sol[c_sol[i]]
        intersection_matrix[idx_ref, idx_sol] += 1

    row_ind, col_ind = linear_sum_assignment(intersection_matrix, maximize=True)
    matrix_mapping = {sol_idx: ref_idx for ref_idx, sol_idx in zip(row_ind, col_ind)}
    label_mapping = {
        label_sol: matrix_idx_to_label_ref[matrix_mapping[idx]]
        for idx, label_sol in matrix_idx_to_label_sol.items()
        if idx in matrix_mapping
    }
    return np.array([label_mapping.get(label, label) for label in c_sol])

# =====================================================================
# 2. BOXPLOTS DE FITNESS
#    → Excluye BL_OPTIMIZADO (no aporta info nueva en fitness)
#    → Incluye EXTRA
# =====================================================================
def plot_fitness_boxplots():
    print("Generando boxplots de Fitness...")
    fnames = glob.glob("../resultados/resultados_*.csv")

    for fname in fnames:
        df = pd.read_csv(fname)
        if 'alg' not in df.columns:
            continue

        df['alg'] = df['alg'].str.upper()
        df = filtrar_algos(df, 'alg', solo_tiempo=False)   # sin BL_OPTIMIZADO
        order = orden_presente(df, 'alg')

        plt.figure(figsize=(9, 6))
        p = sns.boxplot(data=df, x="alg", y="fitness", order=order, legend=False)
        p.set_title(f"Distribución de Fitness - {os.path.basename(fname)}", fontsize=12)
        p.set(xlabel="Algoritmo", ylabel="Fitness")

        foutput = fname.replace(".csv", "_fitness.png")
        plt.savefig(foutput, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Boxplot Fitness guardado: {foutput}")

# =====================================================================
# 3. MÉTRICAS DETALLADAS (Boxplots 2x2)
#    → BL_OPTIMIZADO solo en el subplot de Tiempo
#    → EXTRA en todos los subplots
# =====================================================================
def plot_boxplots_ejecuciones():
    print("Generando boxplots de métricas secundarias...")
    archivos = glob.glob("../resultados/ejecuciones_*.csv")

    for archivo in archivos:
        df = pd.read_csv(archivo)
        df['Algoritmo'] = df['Algoritmo'].str.upper()
        nombre_base = os.path.basename(archivo).replace("ejecuciones_", "").replace(".csv", "")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Métricas Detalladas - Dataset: {nombre_base.upper()}",
                     fontsize=16, fontweight='bold')
        metricas = ['Distancia', 'Incumplimientos', 'Evaluaciones', 'Tiempo']
        axes = axes.flatten()

        for i, metrica in enumerate(metricas):
            es_tiempo = (metrica == 'Tiempo')
            # BL_OPTIMIZADO solo aparece en el subplot de Tiempo
            df_plot = filtrar_algos(df, 'Algoritmo', solo_tiempo=es_tiempo)
            order = orden_presente(df_plot, 'Algoritmo')

            sns.boxplot(data=df_plot, x='Algoritmo', y=metrica,
                        order=order, ax=axes[i], legend=False)
            axes[i].set_title(
                metrica + (" (incl. BL_OPTIMIZADO)" if es_tiempo else ""),
                fontsize=11
            )
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=15)

        plt.tight_layout()
        out = f"../resultados/metricas_detalladas_{nombre_base}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Métricas guardadas: {out}")

# =====================================================================
# 4. COMPARATIVA GLOBAL (barras)
#    → BL_OPTIMIZADO solo en el subplot de Tiempo
#    → EXTRA en todos
# =====================================================================
def plot_compact_metrics():
    resumen_csv = "../resultados/resumen_metricas.csv"
    if not os.path.exists(resumen_csv):
        return

    df = pd.read_csv(resumen_csv)
    df['Algoritmo'] = df['Algoritmo'].str.upper()

    metricas = ['Fitness', 'Distancia', 'Incumplimientos', 'Tiempo']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparativa Global por Dataset", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, metrica in enumerate(metricas):
        es_tiempo = (metrica == 'Tiempo')
        df_plot = filtrar_algos(df, 'Algoritmo', solo_tiempo=es_tiempo)
        order_hue = orden_presente(df_plot, 'Algoritmo')

        sns.barplot(data=df_plot, x='Dataset', y=metrica,
                    hue='Algoritmo', hue_order=order_hue, ax=axes[i])
        axes[i].set_title(
            metrica + (" (incl. BL_OPTIMIZADO)" if es_tiempo else ""),
            fontsize=11
        )
        axes[i].tick_params(axis='x', rotation=15)
        axes[i].legend(title="Algoritmo", fontsize='small',
                       title_fontsize='9', loc='best')

    plt.tight_layout()
    plt.savefig("../resultados/comparativa_global.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Comparativa global guardada.")

# =====================================================================
# 5. PANEL PCA
#    → Incluye EXTRA además de RANDOM, GREEDY, BL
# =====================================================================
def plot_combined_clusters_panel(df_all_sols_unmatched, X_pca, dataset, tag, output_img):
    # ← EXTRA añadido al panel PCA
    orden_preferencia = ['RANDOM', 'GREEDY', 'BL', 'EXTRA']
    ejecutados = [a for a in orden_preferencia
                  if a in df_all_sols_unmatched['Algoritmo'].unique()]
    N = len(ejecutados)
    if N == 0:
        return

    fig, axes = plt.subplots(1, N, figsize=(7 * N, 6))
    fig.suptitle(
        f"Particiones Clustering PCA - {dataset.upper()} ({tag}% Restr.)",
        fontsize=18, fontweight='bold', y=1.05
    )
    if N == 1:
        axes = [axes]

    for i, algo in enumerate(ejecutados):
        df_algo = df_all_sols_unmatched[
            df_all_sols_unmatched['Algoritmo'] == algo
        ].copy()
        df_algo['Cluster_ID'] = df_algo['Cluster'].astype(str)

        clusters_encontrados = sorted(df_algo['Cluster'].unique())
        hue_order = [str(c) for c in clusters_encontrados]
        num_clusters = len(clusters_encontrados)

        palette = sns.color_palette("Set2" if num_clusters <= 8 else "tab20", num_clusters)
        color_mapping = {str(c): palette[j] for j, c in enumerate(clusters_encontrados)}

        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1],
            hue='Cluster_ID', hue_order=hue_order,
            data=df_algo, s=65, alpha=0.8, edgecolor='w',
            ax=axes[i], palette=color_mapping
        )
        axes[i].legend(
            title="IDs Clúster", loc='best',
            fontsize='small', title_fontsize='11',
            frameon=True, ncol=1 if num_clusters <= 5 else 2
        )
        axes[i].set_title(f"Algoritmo: {algo}", fontsize=14,
                          fontweight='semibold', pad=10)
        axes[i].set_xlabel("Componente PCA 1")
        axes[i].set_ylabel("Componente PCA 2" if i == 0 else "")

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Panel PCA guardado: {output_img}")

# =====================================================================
# BLOQUE PRINCIPAL
# =====================================================================
if __name__ == '__main__':
    plot_fitness_boxplots()
    plot_compact_metrics()
    plot_boxplots_ejecuciones()

    archivos_solucion = glob.glob("../resultados/solucion_*.csv")
    if archivos_solucion:
        combos = set()
        for f in archivos_solucion:
            p = os.path.basename(f).split("_")
            if len(p) >= 3:
                combos.add((p[1], p[2]))

        for ds, tg in combos:
            path_dat = f"../data/{ds}_set.dat"
            if not os.path.exists(path_dat):
                continue

            X = pd.read_csv(path_dat, header=None).values[:, :-1]
            X_pca = PCA(n_components=2).fit_transform(X)

            sols = glob.glob(f"../resultados/solucion_{ds}_{tg}_*.csv")
            df_all = pd.DataFrame()

            ref_path = next((s for s in sols if "BL_Optimizado" in s), sols[0])
            df_ref = pd.read_csv(ref_path)

            for s_file in sols:
                base = os.path.basename(s_file).replace(".csv", "")
                algo = base.split("_")[-1].upper()
                if "OPTIMIZADO" in base.upper():
                    algo = "BL_OPTIMIZADO"

                df_curr = pd.read_csv(s_file)
                df_curr['Matched_Cluster'] = match_clusters_to_reference(df_curr, df_ref)
                df_curr['Algoritmo'] = algo
                df_all = pd.concat([df_all, df_curr], ignore_index=True)

            plot_combined_clusters_panel(
                df_all, X_pca, ds, tg,
                f"../resultados/pca_{ds}_{tg}.png"
            )

    print("\n[OK] Todas las gráficas generadas en '../resultados/'.")