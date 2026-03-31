#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <map>
#include <chrono>

#include "parproblem.h"
#include "greedy.h"
#include "randomsearch.h"
#include "localsearch.h"
#include "localsearchoptimizado.h"
#include "extra.h"
#include "random.hpp"

using namespace std;

struct DatasetInfo {
    string nombre, file_data, file_const;
    int k;
    string tag;
};

struct Stats {
    double fit_medio = 0, dist_media = 0, inc_medio = 0, tiempo_medio = 0;
    double eval_medias = 0;
};

string to_lower(string data) {
    transform(data.begin(), data.end(), data.begin(), ::tolower);
    return data;
}

vector<string> split(const string &s, char delimitador) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimitador)) tokens.push_back(token);
    return tokens;
}

void imprimir_tabla(const DatasetInfo &d, const map<string, Stats> &resultados_algos) {
    string nombre_dataset = d.nombre;
    nombre_dataset[0] = toupper(nombre_dataset[0]);

    cout << "\n------------------------------------ Resultados: " << nombre_dataset
         << " (" << d.tag << "% Restricciones) ---------------------------------\n\n";

    int w_alg = 17, w_fit = 13, w_dist = 15, w_inc = 14, w_ev = 10, w_time = 16;

    cout << "| " << left << setw(w_alg) << "Algoritmo"
         << " | " << left << setw(w_fit)  << "Fitness"
         << " | " << left << setw(w_dist) << "Distancia"
         << " | " << left << setw(w_inc)  << "Incumpl."
         << " | " << left << setw(w_ev)   << "Evals."
         << " | " << left << setw(w_time) << "Tiempo (s)" << " |\n";
    cout << "| " << string(w_alg, '-')
         << " | " << string(w_fit, '-')
         << " | " << string(w_dist, '-')
         << " | " << string(w_inc, '-')
         << " | " << string(w_ev, '-')
         << " | " << string(w_time, '-') << " |\n";

    vector<string> orden = {"Greedy", "Random", "BL", "BL_Optimizado", "Extra"};
    for (const string &nombre_algo : orden) {
        if (resultados_algos.count(nombre_algo)) {
            const Stats &s = resultados_algos.at(nombre_algo);
            cout << "| " << left << setw(w_alg) << nombre_algo
                 << " | " << left << setw(w_fit)  << fixed << setprecision(6) << s.fit_medio
                 << " | " << left << setw(w_dist) << fixed << setprecision(6) << s.dist_media
                 << " | " << left << setw(w_inc)  << fixed << setprecision(6) << s.inc_medio
                 << " | " << left << setw(w_ev)   << (int)s.eval_medias
                 << " | " << left << setw(w_time) << fixed << setprecision(6) << s.tiempo_medio << " |\n";
        }
    }
}

int main(int argc, char *argv[]) {
    auto print_help = []() {
        cout << "\n======================================================================\n";
        cout << "USO ESTANDAR: ./main <algoritmo> [fichero] [semilla] [ejecuciones]\n";
        cout << "----------------------------------------------------------------------\n";
        cout << "  Opciones <algoritmo> : greedy | bl | bl_optimizado | random | extra | all\n";
        cout << "  Opciones [fichero]   : zoo | glass | bupa | all\n";
        cout << "======================================================================\n\n";
    };

    if (argc < 2) {
        cout << "\n[ERROR] No se han proporcionado argumentos.\n";
        print_help();
        return 1;
    }

    string arg_algoritmos = to_lower(argv[1]);
    vector<string> target_algos = split(arg_algoritmos, ',');
    string target_data = (argc >= 3) ? to_lower(argv[2]) : "all";

    vector<DatasetInfo> datasets_a_ejecutar;
    long int semilla = (argc >= 4) ? stol(argv[3]) : 88;
    int num_runs = (argc >= 5) ? stoi(argv[4]) : 10;

    // Carga de datasets predefinidos
    vector<DatasetInfo> predefinidos = {
        {"zoo",   "../data/zoo_set.dat",   "../data/zoo_set_const_15.dat",   7,  "15"},
        {"zoo",   "../data/zoo_set.dat",   "../data/zoo_set_const_30.dat",   7,  "30"},
        {"glass", "../data/glass_set.dat", "../data/glass_set_const_15.dat", 7,  "15"},
        {"glass", "../data/glass_set.dat", "../data/glass_set_const_30.dat", 7,  "30"},
        {"bupa",  "../data/bupa_set.dat",  "../data/bupa_set_const_15.dat",  16, "15"},
        {"bupa",  "../data/bupa_set.dat",  "../data/bupa_set_const_30.dat",  16, "30"}
    };

    for (const auto &d : predefinidos) {
        if (target_data == "all" || to_lower(d.nombre) == target_data)
            datasets_a_ejecutar.push_back(d);
    }

    for (const auto &d : datasets_a_ejecutar) {
        string filename = "../resultados/" + d.nombre + "_" + d.tag + ".csv";
        ofstream f_init(filename);
        f_init << "alg,fitness\n";
        f_init.close();

        map<string, Stats> resultados_dataset;

        vector<pair<string, MH<int>*>> todos_algos = {
            {"Random", new RandomSearch<int>()},
            {"Greedy", new GreedySearch()},
            {"BL", new LocalSearch()},
            {"BL_Optimizado", new LocalSearchOptimizado()},
            {"Extra", new Extra()}
        };

        for (auto& [nombre_algo, ptr_algo] : todos_algos) {
            bool ejecutar = false;
            for (const string &t : target_algos)
                if (t == "all" || t == to_lower(nombre_algo)) { ejecutar = true; break; }

            if (ejecutar) {
                Stats s;
                for (int run = 0; run < num_runs; ++run) {
                    ParProblem problem(d.k);
                    if (problem.loadData(d.file_data, d.file_const)) {
                        Random::seed(semilla + run);
                        problem.setSeed(semilla + run);
                        
                        auto t_start = chrono::high_resolution_clock::now();
                        ResultMH<int> res = ptr_algo->optimize(problem, 100000);
                        auto t_end = chrono::high_resolution_clock::now();

                        double tiempo = chrono::duration<double>(t_end - t_start).count();
                        
                        // Acumular estadísticas para la tabla
                        s.fit_medio += res.fitness;
                        s.dist_media += problem.calculateDeviation(res.solution);
                        s.inc_medio += (double)problem.countViolations(res.solution) / (double)problem.getNumRestricciones();
                        s.eval_medias += res.evaluations;
                        s.tiempo_medio += tiempo;

                        // Guardar para boxplot
                        ofstream f_app(filename, ios::app);
                        f_app << nombre_algo << "," << res.fitness << "\n";
                    }
                }
                // Promediar
                s.fit_medio /= num_runs;
                s.dist_media /= num_runs;
                s.inc_medio /= num_runs;
                s.eval_medias /= num_runs;
                s.tiempo_medio /= num_runs;
                resultados_dataset[nombre_algo] = s;
            }
            delete ptr_algo;
        }
        imprimir_tabla(d, resultados_dataset);
    }

    return 0;
}