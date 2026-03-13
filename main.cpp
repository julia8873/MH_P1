#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <map>
#include <sys/stat.h>

#include "parproblem.h"
#include "greedy.h"
#include "randomsearch.h"
#include "localsearch.h"
#include "localsearchnooptimization.h"
#include "random.hpp"

using namespace std;

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

// Estructura para acumular los promedios de todos los datasets [cite: 100]
struct GlobalSummary {
    double fit_total = 0, dist_total = 0, inc_total = 0, tiempo_total = 0;
    double eval_total = 0;
    int datasets_contados = 0;
};

void imprimir_mensaje(char* program_name) {
    cout << "============================================================" << endl;
    cout << "COMO EJECUTAR" << endl;
    cout << "============================================================" << endl;
    cout << "Uso 1 (Automatico): " << program_name << " [semilla]" << endl;
    cout << "Uso 2 (Manual): " << program_name << " <semilla> <datos> <const> <k> <etiqueta>" << endl;
    cout << "============================================================\n" << endl;
}

// Se pasa el mapa global por referencia para acumular resultados [cite: 156]
void run_experiments(const DatasetInfo& d, long int base_seed, map<string, GlobalSummary>& global_accum) {
    cout << "\n[DEBUG] Cargando: " << d.nombre << " (" << d.tag << "% restricciones)... " << flush;
    ParProblem problem(d.k);
    problem.loadData(d.file_data, d.file_const);
    cout << "OK." << endl;

    vector<pair<string, MH<int>*>> algoritmos = { 
        {"Random", new RandomSearch<int>()}, 
        {"Greedy", new GreedySearch()}, 
        {"BL", new LocalSearch()},
        {"BL_NoOpt", new LocalSearchNoOptimization()}
    };

    cout << "\nResultados para: " << d.nombre << " (" << d.tag << "% restricciones)" << endl;
    cout << left << setw(12) << "Algoritmo" << setw(12) << "Fitness" << setw(12) << "Distancia" 
         << setw(18) << "Incumplimiento" << setw(15) << "Evaluaciones" << "Tiempo (s)" << endl;
    cout << string(95, '-') << endl;

    for (auto& algo_pair : algoritmos) {
        AlgoStats s;
        string nombre = algo_pair.first;
        vector<double> historial_tiempos;

        for (int run = 0; run < 50; ++run) {
            Random::seed(base_seed + run);
            problem.setSeed(base_seed + run); 
            
            auto t_start = chrono::high_resolution_clock::now();
            ResultMH<int> res = algo_pair.second->optimize(problem, 100000); 
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

        s.fit_medio /= 50.0;
        s.dist_media /= 50.0;
        s.inc_medio /= 50.0;
        s.eval_medias /= 50.0;
        s.tiempo_medio = accumulate(historial_tiempos.begin(), historial_tiempos.end(), 0.0) / 50.0;

        global_accum[nombre].fit_total += s.fit_medio;
        global_accum[nombre].dist_total += s.dist_media;
        global_accum[nombre].inc_total += s.inc_medio;
        global_accum[nombre].eval_total += s.eval_medias;
        global_accum[nombre].tiempo_total += s.tiempo_medio;
        global_accum[nombre].datasets_contados++;

        cout << left << setw(12) << nombre 
             << setw(12) << fixed << setprecision(4) << s.fit_medio 
             << setw(12) << s.dist_media 
             << setw(18) << s.inc_medio 
             << setw(15) << (int)s.eval_medias 
             << s.tiempo_medio << endl;

        string folder = "results_" + d.nombre + "_" + d.tag;
        mkdir(folder.c_str(), 0777);
        
        // Exportar fitness para boxplots
        ofstream f_fit(folder + "/fitness_" + nombre + ".csv");
        for(double v : s.historial_fitness) f_fit << v << "\n";
        
        // Exportar todos los tiempos para boxplots de tiempo
        ofstream f_time(folder + "/times_" + nombre + ".csv");
        for(double t : historial_tiempos) f_time << t << "\n";
        
        // Exportar mejor solución
        ofstream f_sol(folder + "/best_sol_" + nombre + ".csv");
        for(int c : s.mejor_solucion) f_sol << c << "\n";
    }
    for(auto& a : algoritmos) delete a.second;
}

int main(int argc, char *argv[]) {
    imprimir_mensaje(argv[0]);

    long int base_seed = 2026;
    vector<DatasetInfo> datasets;
    map<string, GlobalSummary> global_results;

    if (argc >= 6) {
        base_seed = stol(argv[1]);
        datasets.push_back({argv[5], argv[2], argv[3], stoi(argv[4]), "manual"});
    } else {
        if (argc >= 2) base_seed = stol(argv[1]);
        datasets = {
            {"zoo", "../data/zoo_set.dat", "../data/zoo_set_const_15.dat", 7, "15"},
            {"zoo", "../data/zoo_set.dat", "../data/zoo_set_const_30.dat", 7, "30"},
            {"glass", "../data/glass_set.dat", "../data/glass_set_const_15.dat", 7, "15"},
            {"glass", "../data/glass_set.dat", "../data/glass_set_const_30.dat", 7, "30"},
            {"bupa", "../data/bupa_set.dat", "../data/bupa_set_const_15.dat", 16, "15"},
            {"bupa", "../data/bupa_set.dat", "../data/bupa_set_const_30.dat", 16, "30"}
        };
    }

    for (const auto& d : datasets) {
        run_experiments(d, base_seed, global_results);
    }

    cout << "============================================================" << endl;
    cout << "TABLA GLOBAL RESUMEN (Media de todos los casos)" << endl;
    cout << "============================================================" << endl;
    cout << left << setw(12) << "Algoritmo" << setw(12) << "Fitness" << setw(12) << "Distancia" 
         << setw(18) << "Incumplimiento" << setw(15) << "Evaluaciones" << "Tiempo (s)" << endl;
    cout << string(85, '-') << endl;

    ofstream f_global("global_results.csv");
    f_global << "Algoritmo,Fitness,Distancia,Incumplimiento,Evaluaciones,Tiempo\n";

    for (auto const& [nombre, summary] : global_results) {
        double n = summary.datasets_contados;
        double f_med = summary.fit_total / n;
        double d_med = summary.dist_total / n;
        double i_med = summary.inc_total / n;
        double e_med = summary.eval_total / n;
        double t_med = summary.tiempo_total / n;

        cout << left << setw(12) << nombre 
             << setw(12) << fixed << setprecision(4) << f_med 
             << setw(12) << d_med 
             << setw(18) << i_med 
             << setw(15) << (int)e_med 
             << t_med << endl;

        f_global << nombre << "," << f_med << "," << d_med << "," << i_med << "," << (int)e_med << "," << t_med << "\n";
    }
    f_global.close();

    return 0;
}