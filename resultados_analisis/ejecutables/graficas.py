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
ALGOS_TODAS = ['GREEDY', 'RANDOM', 'BL', 'EXTRA']
ALGOS_SOLO_TIEMPO = ['BL_OPTIMIZADO']
ORDEN_ALGOS = ['GREEDY', 'RANDOM', 'BL', 'BL_OPTIMIZADO', 'EXTRA']

def filtrar_algos(df, col_algo, solo_tiempo=False):
    if solo_tiempo:
        permitidos = ALGOS_TODAS + ALGOS_SOLO_TIEMPO
    else:
        permitidos = ALGOS_TODAS
    return df[df[col_algo].isin(permitidos)]

def orden_presente(df, col_algo):
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
# =====================================================================
def plot_fitness_boxplots():
    print("Generando boxplots de Fitness (GREEDY, RANDOM, BL en la misma gráfica)...")
    archivos = glob.glob("../resultados/ejecuciones_*.csv")

    algos_permitidos = ['GREEDY', 'RANDOM', 'BL']

    for archivo in archivos:
        df = pd.read_csv(archivo)
        
        if 'Algoritmo' not in df.columns or 'Fitness' not in df.columns:
            continue

        df['Algoritmo'] = df['Algoritmo'].str.upper()
        df = df[df['Algoritmo'].isin(algos_permitidos)]
        
        if df.empty:
            continue
            
        order = [a for a in algos_permitidos if a in df['Algoritmo'].unique()]
        nombre_base = os.path.basename(archivo).replace("ejecuciones_", "").replace(".csv", "")

        plt.figure(figsize=(10, 8))
        
        sns.boxplot(data=df, x="Algoritmo", y="Fitness", order=order, color="tab:blue", width=0.6)
        sns.despine()

        plt.title(f"Distribución de Fitness - {nombre_base.upper()}", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Algoritmo", fontsize=14, fontweight='semibold')
        plt.ylabel("Fitness", fontsize=14, fontweight='semibold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        foutput = f"../resultados/boxplot_fitness_{nombre_base}.png"
        plt.savefig(foutput, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Boxplot Fitness unificado guardado: {foutput}")

# =====================================================================
# 3. MÉTRICAS DETALLADAS
# =====================================================================
def plot_boxplots_ejecuciones():
    print("Generando boxplots de métricas secundarias...")
    archivos = glob.glob("../resultados/ejecuciones_*.csv")

    for archivo in archivos:
        df = pd.read_csv(archivo)
        df['Algoritmo'] = df['Algoritmo'].str.upper()
        nombre_base = os.path.basename(archivo).replace("ejecuciones_", "").replace(".csv", "")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f"Métricas Detalladas - Dataset: {nombre_base.upper()}",
                     fontsize=20, fontweight='bold', y=1.02)
        
        metricas = ['Distancia', 'Incumplimientos', 'Evaluaciones', 'Tiempo']
        axes = axes.flatten()

        for i, metrica in enumerate(metricas):
            es_tiempo = (metrica == 'Tiempo')
            df_plot = filtrar_algos(df, 'Algoritmo', solo_tiempo=es_tiempo)
            order = orden_presente(df_plot, 'Algoritmo')

            sns.boxplot(data=df_plot, x='Algoritmo', y=metrica,
                        order=order, ax=axes[i], legend=False, width=0.6)
            
            axes[i].set_title(
                metrica + (" (incl. BL_OPTIMIZADO)" if es_tiempo else ""),
                fontsize=15, fontweight='semibold', pad=10
            )
            axes[i].set_xlabel('')
            axes[i].set_ylabel(metrica, fontsize=13)
            axes[i].tick_params(axis='x', rotation=15, labelsize=12)
            axes[i].tick_params(axis='y', labelsize=12)

        plt.tight_layout()
        out = f"../resultados/metricas_detalladas_{nombre_base}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Métricas guardadas: {out}")

# =====================================================================
# 4. COMPARATIVA GLOBAL (Con valores numéricos)
# =====================================================================
def plot_compact_metrics():
    print("Generando comparativa global con todos los datos y valores numéricos...")
    archivos = glob.glob("../resultados/ejecuciones_*.csv")
    
    if not archivos:
        print("[!] No se encontraron archivos de ejecuciones.")
        return

    lista_df = []
    for f in archivos:
        df_temp = pd.read_csv(f)
        dataset_name = os.path.basename(f).replace("ejecuciones_", "").replace(".csv", "")
        df_temp['Dataset'] = dataset_name
        lista_df.append(df_temp)
        
    df = pd.concat(lista_df, ignore_index=True)
    df['Algoritmo'] = df['Algoritmo'].str.upper()

    metricas = ['Fitness', 'Distancia', 'Incumplimientos', 'Tiempo']
    fig, axes = plt.subplots(2, 2, figsize=(18, 14)) # Un poco más grande para que quepan los textos
    fig.suptitle("Comparativa Global por Dataset (Media de todas las ejecuciones)", fontsize=18, fontweight='bold', y=1.02)
    axes = axes.flatten()

    for i, metrica in enumerate(metricas):
        if metrica not in df.columns:
            continue
            
        es_tiempo = (metrica == 'Tiempo')
        df_plot = filtrar_algos(df, 'Algoritmo', solo_tiempo=es_tiempo)
        order_hue = orden_presente(df_plot, 'Algoritmo')

        sns.barplot(data=df_plot, x='Dataset', y=metrica,
                    hue='Algoritmo', hue_order=order_hue, ax=axes[i], 
                    errorbar=None)
                    
        # --- NUEVO: Añadir los valores numéricos a las barras ---
        for container in axes[i].containers:
            # Añadimos los valores rotados 90 grados para que no se pisen
            axes[i].bar_label(container, fmt='%.1f', fontsize=9, padding=4, rotation=90)
        
        # Ampliamos el margen superior para que el texto no se corte
        axes[i].margins(y=0.25) 
        # --------------------------------------------------------

        axes[i].set_title(
            metrica + (" (incl. BL_OPTIMIZADO)" if es_tiempo else ""),
            fontsize=14
        )
        axes[i].tick_params(axis='x', rotation=15, labelsize=11)
        axes[i].tick_params(axis='y', labelsize=11)
        axes[i].set_xlabel('')
        axes[i].set_ylabel(metrica, fontsize=12)
        axes[i].legend(title="Algoritmo", fontsize='10',
                       title_fontsize='11', loc='best')

    plt.tight_layout()
    plt.savefig("../resultados/comparativa_global.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Comparativa global guardada usando todos los datos (con valores).")

def plot_convergencia():
    print("Generando gráficos de convergencia...")
    archivos = glob.glob("../resultados/convergencia_*.csv")

    grupos = {}
    for f in archivos:
        base = os.path.basename(f).replace("convergencia_", "").replace(".csv", "")
        parts = base.split("_")
        dataset_tag = "_".join(parts[:2])
        algo = "_".join(parts[2:]).upper()
        
        if dataset_tag not in grupos: grupos[dataset_tag] = []
        grupos[dataset_tag].append((algo, f))

    for key, lista in grupos.items():
        plt.figure(figsize=(10, 6))
        x_common = np.linspace(1, 100000, 2000)

        for algo, f in lista:
            df = pd.read_csv(f)
            if df.empty: continue

            run_curves = []
            for run_id in df['Run'].unique():
                df_run = df[df['Run'] == run_id].sort_values('Iter')
                x_run = df_run['Iter'].values
                y_run = df_run['Fitness'].values

                if len(x_run) == 1:
                    x_run = np.append(x_run, 100000)
                    y_run = np.append(y_run, y_run[0])
                
                y_interp = np.interp(x_common, x_run, y_run)
                y_interp = np.minimum.accumulate(y_interp)
                run_curves.append(y_interp)

            if not run_curves: continue

            y_mean = np.mean(run_curves, axis=0)
            plt.plot(x_common, y_mean, label=algo, linewidth=2.0)

        plt.title(f"Curva de Convergencia Media - {key.upper()}", fontsize=15, fontweight='bold')
        plt.xlabel("Evaluaciones de Fitness", fontsize=12)
        plt.ylabel("Fitness (Menor es mejor)", fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        
        out = f"../resultados/convergencia_{key}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Gráfica de convergencia lista: {out}")

# =====================================================================
# 7. ESTABILIDAD (Con valores numéricos)
# =====================================================================
def plot_estabilidad():
    print("Generando gráficos de estabilidad con valores numéricos...")

    archivos = glob.glob("../resultados/ejecuciones_*.csv")
    datos = []

    for f in archivos:
        df_exec = pd.read_csv(f)

        if 'Algoritmo' not in df_exec.columns or 'Fitness' not in df_exec.columns:
            continue

        df_exec['Algoritmo'] = df_exec['Algoritmo'].str.upper()
        dataset = os.path.basename(f).replace("ejecuciones_", "").replace(".csv", "")

        for algo in df_exec['Algoritmo'].unique():
            vals = df_exec[df_exec['Algoritmo'] == algo]['Fitness']
            if len(vals) == 0:
                continue
            datos.append({
                'Dataset': dataset,
                'Algoritmo': algo,
                'StdFitness': np.std(vals)
            })

    if len(datos) == 0:
        print("[!] No hay datos para estabilidad, se omite gráfica.")
        return

    df_std = pd.DataFrame(datos)

    if 'Dataset' not in df_std.columns:
        print("[!] Error interno: falta columna Dataset.")
        return

    plt.figure(figsize=(14, 8)) # Hacemos la gráfica un poco más ancha
    ax = sns.barplot(data=df_std, x='Dataset', y='StdFitness', hue='Algoritmo')
    
    # --- NUEVO: Añadir los valores numéricos a las barras ---
    for container in ax.containers:
        # Aquí usamos 4 decimales porque las desviaciones pueden ser muy pequeñas
        ax.bar_label(container, fmt='%.4f', fontsize=10, padding=4, rotation=90)
    
    # Ampliamos el margen superior para que quepan las etiquetas
    ax.margins(y=0.2)
    # --------------------------------------------------------

    plt.title("Estabilidad (Desviación estándar del fitness)", fontsize=16, fontweight='bold')
    plt.xlabel("Dataset", fontsize=13)
    plt.ylabel("Desviación Estándar", fontsize=13)
    plt.xticks(rotation=15, fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(title="Algoritmo", fontsize='11', title_fontsize='12')

    out = "../resultados/estabilidad.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Estabilidad guardada: {out}")

# =====================================================================
# 5. PANEL PCA
# =====================================================================
def plot_combined_clusters_panel(df_all_sols_unmatched, X_pca, dataset, tag, output_img):
    orden_preferencia = ['RANDOM', 'GREEDY', 'BL', 'EXTRA']
    ejecutados = [a for a in orden_preferencia
                  if a in df_all_sols_unmatched['Algoritmo'].unique()]
    N = len(ejecutados)
    if N == 0:
        return

    fig, axes = plt.subplots(1, N, figsize=(8 * N, 7))
    fig.suptitle(
        f"Particiones Clustering PCA - {dataset.upper()} ({tag}% Restr.)",
        fontsize=20, fontweight='bold', y=1.05
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
            data=df_algo, s=80, alpha=0.8, edgecolor='w',
            ax=axes[i], palette=color_mapping
        )
        axes[i].legend(
            title="IDs Clúster", loc='best',
            fontsize='10', title_fontsize='12',
            frameon=True, ncol=1 if num_clusters <= 5 else 2
        )
        axes[i].set_title(f"Algoritmo: {algo}", fontsize=16,
                          fontweight='semibold', pad=12)
        axes[i].set_xlabel("Componente PCA 1", fontsize=12)
        axes[i].set_ylabel("Componente PCA 2" if i == 0 else "", fontsize=12)
        axes[i].tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Panel PCA guardado: {output_img}")

# =====================================================================
# 8. COMPARATIVA GLOBAL DE TIEMPOS (MEDIA) - BL vs BL_OPTIMIZADO
# =====================================================================
def plot_tiempos_media_global():
    print("Generando comparativa global de tiempos medios (BL vs BL_OPTIMIZADO)...")
    archivos = glob.glob("../resultados/ejecuciones_*.csv")
    
    if not archivos:
        print("[!] No se encontraron archivos de ejecuciones para la gráfica de tiempos.")
        return

    datos = []
    algos_comparar = ['BL', 'BL_OPTIMIZADO']
    datasets_objetivo = ['zoo', 'glass', 'bupa']

    for f in archivos:
        dataset_name = os.path.basename(f).replace("ejecuciones_", "").replace(".csv", "")
        
        # Filtramos solo los datasets que nos interesan
        if not any(target in dataset_name.lower() for target in datasets_objetivo):
            continue

        df = pd.read_csv(f)
        if 'Algoritmo' not in df.columns or 'Tiempo' not in df.columns:
            continue

        df['Algoritmo'] = df['Algoritmo'].str.upper()
        df_bl = df[df['Algoritmo'].isin(algos_comparar)].copy()
        
        if df_bl.empty:
            continue

        df_bl['Dataset'] = dataset_name
        datos.append(df_bl)

    if not datos:
        print("[!] No hay datos de BL o BL_OPTIMIZADO para los datasets indicados.")
        return

    df_total = pd.concat(datos, ignore_index=True)

    # Imprimir resumen de SUMA y MEDIA por consola
    resumen = df_total.groupby(['Dataset', 'Algoritmo'])['Tiempo'].agg(['sum', 'mean']).reset_index()
    print("\n--- RESUMEN DE TIEMPOS (Segundos) ---")
    print(resumen.to_string(index=False))
    print("-------------------------------------\n")

    # Crear la gráfica
    plt.figure(figsize=(14, 7))
    
    # barplot calcula la media ('mean') por defecto al agrupar
    ax = sns.barplot(
        data=df_total, 
        x='Dataset', 
        y='Tiempo', 
        hue='Algoritmo',
        palette=['tab:blue', 'tab:orange'],
        errorbar=None # Quitamos las líneas de error para que se vean bien los números
    )
    
    # Añadir los valores numéricos (Media) encima de cada barra
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=10, padding=4, rotation=45)
        
    ax.margins(y=0.3) # Ampliamos el margen superior para que no se corten los números
    
    plt.title("Tiempo Medio de Ejecución: BL vs BL_OPTIMIZADO", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Dataset", fontsize=14, fontweight='semibold')
    plt.ylabel("Tiempo Medio (segundos)", fontsize=14, fontweight='semibold')
    plt.xticks(rotation=15, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Algoritmo", fontsize=11, title_fontsize=12)
    
    out = "../resultados/tiempos_media_bl_global.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfica global de tiempos medios guardada: {out}")

# =====================================================================
# 11. TABLA BONITA DE RESUMEN GLOBAL (EXTRA VS RESTO)
# =====================================================================
def print_resumen_global_mejoras():
    archivos = glob.glob("../resultados/ejecuciones_*.csv")
    if not archivos:
        return

    lista_df = []
    for f in archivos:
        df_temp = pd.read_csv(f)
        if 'Algoritmo' not in df_temp.columns or 'Fitness' not in df_temp.columns:
            continue
        dataset_name = os.path.basename(f).replace("ejecuciones_", "").replace(".csv", "")
        df_temp['Dataset'] = dataset_name
        lista_df.append(df_temp)
        
    if not lista_df:
        return

    df = pd.concat(lista_df, ignore_index=True)
    df['Algoritmo'] = df['Algoritmo'].str.upper()

    # 1. Sacamos la media de fitness por Dataset y Algoritmo
    resumen = df.groupby(['Dataset', 'Algoritmo'])['Fitness'].mean().unstack()

    if 'EXTRA' not in resumen.columns:
        print("[!] No se puede generar el resumen: Falta el algoritmo EXTRA.")
        return

    algos_comparar = [a for a in ['RANDOM', 'GREEDY', 'BL'] if a in resumen.columns]

    # 2. Dibujamos la cabecera de la tabla bonita
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " RESUMEN GLOBAL: RENDIMIENTO DE 'EXTRA' VS RESTO ".center(68) + "║")
    print("╠" + "═"*22 + "╦" + "═"*22 + "╦" + "═"*22 + "╣")
    print("║" + " Comparativa ".center(22) + "║" + " Mejora Media (%) ".center(22) + "║" + " Veredicto Final ".center(22) + "║")
    print("╠" + "═"*22 + "╬" + "═"*22 + "╬" + "═"*22 + "╣")

    # 3. Calculamos e imprimimos las filas
    for algo in algos_comparar:
        mejoras_por_dataset = []
        
        for dataset, row in resumen.iterrows():
            val_extra = row.get('EXTRA', np.nan)
            val_algo = row.get(algo, np.nan)
            
            if pd.notna(val_extra) and pd.notna(val_algo) and val_algo != 0:
                # Fórmula: % Mejora = ((Viejo - Nuevo) / Viejo) * 100
                pct = ((val_algo - val_extra) / val_algo) * 100
                mejoras_por_dataset.append(pct)
        
        if mejoras_por_dataset:
            media_mejora = np.mean(mejoras_por_dataset)
            
            # Formatear el texto de las columnas
            col_comparativa = f" EXTRA vs {algo}".ljust(22)
            
            signo = "+" if media_mejora > 0 else ""
            col_porcentaje = f" {signo}{media_mejora:.2f} %".center(22)
            
            if media_mejora > 0.5:
                veredicto = " MEJORA ".center(22)
            elif media_mejora < -0.5:
                veredicto = " EMPEORA ".center(22)
            else:
                veredicto = " ⚖️ EMPATE TÉCNICO ".center(22)
                
            print(f"║{col_comparativa}║{col_porcentaje}║{veredicto}║")

    # 4. Pie de tabla
    print("╚" + "═"*22 + "╩" + "═"*22 + "╩" + "═"*22 + "╝")
    print("  * Nota: Valores positivos (+) indican que EXTRA tiene menor fitness de media")

# =====================================================================
# 10. GRÁFICA DE MEJORA/EMPEORAMIENTO RESPECTO A BL
# =====================================================================
def plot_mejora_vs_bl():
    print("Generando gráfica de mejora porcentual respecto a BL...")
    archivos = glob.glob("../resultados/ejecuciones_*.csv")
    
    if not archivos:
        print("[!] No se encontraron archivos para la comparativa contra BL.")
        return

    lista_df = []
    for f in archivos:
        df_temp = pd.read_csv(f)
        if 'Algoritmo' not in df_temp.columns or 'Fitness' not in df_temp.columns:
            continue
        dataset_name = os.path.basename(f).replace("ejecuciones_", "").replace(".csv", "")
        df_temp['Dataset'] = dataset_name
        lista_df.append(df_temp)
        
    if not lista_df:
        return

    df = pd.concat(lista_df, ignore_index=True)
    df['Algoritmo'] = df['Algoritmo'].str.upper()

    # Calculamos la media de fitness por dataset y algoritmo
    resumen = df.groupby(['Dataset', 'Algoritmo'])['Fitness'].mean().unstack()

    if 'BL' not in resumen.columns:
        print("[!] No se ha ejecutado el algoritmo BL. No se puede generar la comparativa.")
        return

    # Preparar datos: % Mejora = ((Fitness_BL - Fitness_Algo) / Fitness_BL) * 100
    # (Asumiendo que un MENOR fitness es MEJOR)
    datos_mejora = []
    # Comparamos EXTRA, GREEDY y RANDOM contra BL (excluimos BL_OPTIMIZADO si es solo para tiempos)
    algos_a_comparar = [a for a in ORDEN_ALGOS if a in resumen.columns and a not in ['BL', 'BL_OPTIMIZADO']]

    for dataset, row in resumen.iterrows():
        bl_val = row.get('BL', np.nan)
        if pd.isna(bl_val) or bl_val == 0:
            continue
            
        for algo in algos_a_comparar:
            algo_val = row.get(algo, np.nan)
            if not pd.isna(algo_val):
                mejora = ((bl_val - algo_val) / bl_val) * 100
                datos_mejora.append({
                    'Dataset': dataset,
                    'Algoritmo': algo,
                    'Mejora_Pct': mejora
                })

    if not datos_mejora:
        print("[!] No hay datos suficientes para comparar contra BL.")
        return

    df_mejora = pd.DataFrame(datos_mejora)

    # Crear la gráfica
    plt.figure(figsize=(14, 8))
    
    # Paleta personalizada: EXTRA en verde o rojo para destacar, los demás en gris/azul
    ax = sns.barplot(data=df_mejora, x='Dataset', y='Mejora_Pct', hue='Algoritmo')

    # Añadir línea del 0 (representa a BL)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Búsqueda Local (Base)')

    # Añadir los valores numéricos con signo (+ o -)
    for container in ax.containers:
        # Extraemos el valor para formatearlo correctamente
        labels = [f"{'+' if v > 0 else ''}{v:.2f}%" if not pd.isna(v) else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, fontsize=10, padding=4, rotation=90)

    # Formato general
    y_max = df_mejora['Mejora_Pct'].max()
    y_min = df_mejora['Mejora_Pct'].min()
    # Ampliar márgenes en Y para que el texto rotado quepa bien
    plt.ylim(y_min - abs(y_min)*0.4 - 5, y_max + abs(y_max)*0.4 + 5)

    plt.title("Evolución del Fitness respecto a Búsqueda Local (BL)", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Dataset", fontsize=14, fontweight='semibold')
    plt.ylabel("% de Mejora (Valores > 0 son MEJORES que BL)", fontsize=14, fontweight='semibold')
    plt.xticks(rotation=15, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Arreglar la leyenda para que no se pise
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title="Algoritmo", fontsize=11, title_fontsize=12, loc='best')

    out = "../resultados/mejora_vs_bl.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfica de mejora respecto a BL guardada: {out}")
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

    plot_convergencia()
    plot_estabilidad()
    plot_tiempos_media_global()
    
    # Llamada a la nueva función de impresión por consola
    plot_mejora_vs_bl()
    print_resumen_global_mejoras()
    
    print("\n[OK] Todas las gráficas generadas en '../resultados/'.")