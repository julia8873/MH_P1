#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <map>
#include <sys/stat.h>
#include <algorithm>

// Headers del proyecto
#include "parproblem.h"
#include "greedy.h"
#include "randomsearch.h"
#include "localsearch.h"
#include "localsearchnooptimization.h"
#include "random.hpp"

using namespace std;

// --- Estructuras de Datos ---
struct DatasetInfo {
    string nombre;
    string file_data;
    string file_const;
    int k;
    string tag; 
};

struct AlgoStats {
    double fit_medio = 0, dist_media = 0, inc_medio = 0, tiempo_medio = 0;
    double eval_medias = 0;
    vector<double> historial_fitness;
    vector<int> mejor_solucion;
    double mejor_fitness = 1e18;
};

struct GlobalSummary {
    double fit_total = 0, dist_total = 0, inc_total = 0, tiempo_total = 0;
    double eval_total = 0;
    int datasets_contados = 0;
};

// --- Funciones Auxiliares ---
string to_lower(string data) {
    transform(data.begin(), data.end(), data.begin(), ::tolower);
    return data;
}

void imprimir_cabecera() {
    cout << left << setw(12) << "Algoritmo" 
         << setw(14) << "Fitness" 
         << setw(14) << "Distancia" 
         << setw(18) << "Incumplimiento" 
         << setw(15) << "Evaluaciones" 
         << "Tiempo (s)" << endl;
    cout << string(90, '-') << endl;
}

void run_experiments(const DatasetInfo& d, MH<int>* algo, string nombre_algo, long int base_seed, int num_runs, map<string, GlobalSummary>& global_accum) {
    ParProblem problem(d.k);
    if (!problem.loadData(d.file_data, d.file_const)) return;

    AlgoStats s;
    vector<double> historial_tiempos;

    for (int run = 0; run < num_runs; ++run) {
        Random::seed(base_seed + run);
        problem.setSeed(base_seed + run); 
        
        auto t_start = chrono::high_resolution_clock::now();
        ResultMH<int> res = algo->optimize(problem, 100000); 
        auto t_end = chrono::high_resolution_clock::now();

        chrono::duration<double> diff = t_end - t_start;
        historial_tiempos.push_back(diff.count());
        
        s.fit_medio += res.fitness;
        s.dist_media += problem.calculateDeviation(res.solution); 
        s.inc_medio += (double)problem.countViolations(res.solution) / (double)problem.getNumRestricciones();
        s.eval_medias += res.evaluations;
        s.historial_fitness.push_back(res.fitness);

        if (res.fitness < s.mejor_fitness) {
            s.mejor_fitness = res.fitness;
            s.mejor_solucion = res.solution;
        }
    }

    // Cálculos de medias
    s.fit_medio /= (double)num_runs;
    s.dist_media /= (double)num_runs;
    s.inc_medio /= (double)num_runs;
    s.eval_medias /= (double)num_runs;
    s.tiempo_medio = accumulate(historial_tiempos.begin(), historial_tiempos.end(), 0.0) / (double)num_runs;

    // Acumular para el resumen global
    global_accum[nombre_algo].fit_total += s.fit_medio;
    global_accum[nombre_algo].dist_total += s.dist_media;
    global_accum[nombre_algo].inc_total += s.inc_medio;
    global_accum[nombre_algo].eval_total += s.eval_medias;
    global_accum[nombre_algo].tiempo_total += s.tiempo_medio;
    global_accum[nombre_algo].datasets_contados++;

    // --- IMPRESIÓN DE FILA EN TIEMPO REAL ---
    cout << left << setw(12) << nombre_algo 
         << setw(14) << fixed << setprecision(4) << s.fit_medio 
         << setw(14) << s.dist_media 
         << setw(18) << s.inc_medio 
         << setw(15) << (int)s.eval_medias 
         << fixed << setprecision(4) << s.tiempo_medio << endl;

    // Exportar resultados
    string folder = "results_" + d.nombre + "_" + d.tag;
    mkdir(folder.c_str(), 0777);
    ofstream f_fit(folder + "/fitness_" + nombre_algo + ".csv");
    for(double v : s.historial_fitness) f_fit << v << "\n";
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "\nUso: " << argv[0] << " <algoritmo|all> [semilla] [ejecuciones]" << endl;
        cout << "Ejemplo: " << argv[0] << " all 88 50\n" << endl;
        return 1;
    }

    string target = to_lower(argv[1]);
    long int base_seed = (argc >= 3) ? stol(argv[2]) : 88;
    int num_runs = (argc >= 4) ? stoi(argv[3]) : 10;

    vector<DatasetInfo> datasets = {
        {"zoo", "../data/zoo_set.dat", "../data/zoo_set_const_15.dat", 7, "15"},
        {"zoo", "../data/zoo_set.dat", "../data/zoo_set_const_30.dat", 7, "30"},
        {"glass", "../data/glass_set.dat", "../data/glass_set_const_15.dat", 7, "15"},
        {"glass", "../data/glass_set.dat", "../data/glass_set_const_30.dat", 7, "30"},
        {"bupa", "../data/bupa_set.dat", "../data/bupa_set_const_15.dat", 16, "15"},
        {"bupa", "../data/bupa_set.dat", "../data/bupa_set_const_30.dat", 16, "30"}
    };

    map<string, GlobalSummary> global_results;

    for (const auto& d : datasets) {
        cout << "\n>>> DATASET: " << d.nombre << " (" << d.tag << "% restricciones)" << endl;
        imprimir_cabecera();

        map<string, MH<int>*> algos;
        algos["Random"] = new RandomSearch<int>();
        algos["Greedy"] = new GreedySearch();
        algos["BL"] = new LocalSearch();
        algos["BL_NoOpt"] = new LocalSearchNoOptimization();

        for (auto const& [name, ptr] : algos) {
            if (target == "all" || target == to_lower(name)) {
                run_experiments(d, ptr, name, base_seed, num_runs, global_results);
            }
        }

        for (auto& pair : algos) delete pair.second;
        cout << string(90, '=') << endl;
    }

    // --- TABLA RESUMEN FINAL ---
    if (!global_results.empty()) {
        cout << "\n\n" << string(30, ' ') << "RESUMEN GLOBAL (MEDIAS)" << endl;
        imprimir_cabecera();
        for (auto const& [nombre, summary] : global_results) {
            double n = summary.datasets_contados;
            cout << left << setw(12) << nombre 
                 << setw(14) << fixed << setprecision(4) << (summary.fit_total / n)
                 << setw(14) << (summary.dist_total / n) 
                 << setw(18) << (summary.inc_total / n) 
                 << setw(15) << (int)(summary.eval_total / n) 
                 << (summary.tiempo_total / n) << endl;
        }
    }

    return 0;
}