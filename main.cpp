#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <map>
#include <algorithm>
#include <iomanip>
#include <numeric>

#include "parproblem.h"
#include "greedy.h"
#include "randomsearch.h"
#include "localsearch.h"
#include "localsearchoptimizado.h"
#include "random.hpp"

using namespace std;

struct DatasetInfo
{
    string nombre, file_data, file_const;
    int k;
    string tag;
};

struct Stats
{
    double fit_medio = 0, dist_media = 0, inc_medio = 0, tiempo_medio = 0;
    double eval_medias = 0;
};

string to_lower(string data)
{
    transform(data.begin(), data.end(), data.begin(), ::tolower);
    return data;
}

// Devuelve los estadísticos medios para imprimirlos en la tabla
Stats run_experiments(const DatasetInfo &d, MH<int> *algo, string nombre_algo, long int base_seed, int num_runs)
{
    ParProblem problem(d.k);
    if (!problem.loadData(d.file_data, d.file_const))
        return Stats();

    vector<double> historial_fitness, historial_tiempos;
    Stats s;

    for (int run = 0; run < num_runs; ++run)
    {
        Random::seed(base_seed + run);
        problem.setSeed(base_seed + run);

        auto t_start = chrono::high_resolution_clock::now();
        ResultMH<int> res = algo->optimize(problem, 100000);
        auto t_end = chrono::high_resolution_clock::now();

        double tiempo = chrono::duration<double>(t_end - t_start).count();
        historial_tiempos.push_back(tiempo);
        historial_fitness.push_back(res.fitness);

        // Medias
        s.fit_medio += res.fitness;
        s.dist_media += problem.calculateDeviation(res.solution);
        s.inc_medio += (double)problem.countViolations(res.solution) / (double)problem.getNumRestricciones();
        s.eval_medias += res.evaluations;
        s.tiempo_medio += tiempo;
    }

    // para fitness
    ofstream f_box("resultados_" + d.nombre + "_" + d.tag + ".csv", ios::app);
    for (double v : historial_fitness)
        f_box << nombre_algo << "," << v << "\n";

    // para tiempos
    ofstream f_time("tiempos_" + d.nombre + "_" + d.tag + ".csv", ios::app);
    for (double t : historial_tiempos)
        f_time << nombre_algo << "," << t << "\n";

    // medias
    s.fit_medio /= num_runs;
    s.dist_media /= num_runs;
    s.inc_medio /= num_runs;
    s.eval_medias /= num_runs;
    s.tiempo_medio /= num_runs;

    return s;
}

void imprimir_tabla(const DatasetInfo &d, const map<string, Stats> &resultados_algos)
{
    string nombre_dataset = d.nombre;
    nombre_dataset[0] = toupper(nombre_dataset[0]);

    cout << "\n------------------------------------ Resultados:" << nombre_dataset
         << " (" << d.tag << "% Restricciones) ---------------------------------\n\n";

    int w_alg = 17, w_fit = 13, w_dist = 15, w_inc = 14, w_ev = 10, w_time = 16;

    cout << "| " << left << setw(w_alg) << "Algoritmo"
         << " | " << left << setw(w_fit) << "Fitness"
         << " | " << left << setw(w_dist) << "Distancia"
         << " | " << left << setw(w_inc) << "Incumpl."
         << " | " << left << setw(w_ev) << "Evals."
         << " | " << left << setw(w_time) << "Tiempo (s)" << " |\n";
    cout << "| -" << string(w_alg - 1, '-')
         << " | -" << string(w_fit - 2, '-') << "-"
         << " | -" << string(w_dist - 2, '-') << "-"
         << " | -" << string(w_inc - 2, '-') << "-"
         << " | -" << string(w_ev - 2, '-') << "-"
         << " | -" << string(w_time - 2, '-') << "- |\n";

    vector<string> orden = {"Greedy", "Random", "BL", "BL_Optimizado"};

    for (const string &nombre_algo : orden)
    {
        if (resultados_algos.count(nombre_algo))
        {
            const Stats &s = resultados_algos.at(nombre_algo);
            string str_algo = nombre_algo;
            cout << "| " << left << setw(w_alg) << str_algo
                 << " | " << left << setw(w_fit) << fixed << setprecision(6) << s.fit_medio
                 << " | " << left << setw(w_dist) << fixed << setprecision(6) << s.dist_media
                 << " | " << left << setw(w_inc) << fixed << setprecision(6) << s.inc_medio
                 << " | " << left << setw(w_ev) << (int)s.eval_medias
                 << " | " << left << setw(w_time) << fixed << setprecision(6) << s.tiempo_medio << " |\n";
        }
    }
}
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "Uso: " << argv[0] << " <algoritmo|all> [semilla] [ejecuciones]" << endl;
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
        {"bupa", "../data/bupa_set.dat", "../data/bupa_set_const_30.dat", 16, "30"}};

    for (const auto &d : datasets)
    {
        ofstream f_init("resultados_" + d.nombre + "_" + d.tag + ".csv");
        f_init << "alg,fitness\n";
        f_init.close();

        ofstream t_init("tiempos_" + d.nombre + "_" + d.tag + ".csv");
        t_init << "alg,fitness\n";
        t_init.close();

        map<string, MH<int> *> algos = {
            {"Random", new RandomSearch<int>()},
            {"Greedy", new GreedySearch()},
            {"BL", new LocalSearch()},
            {"BL_Optimizado", new LocalSearchOptimizado()}};

        map<string, Stats> resultados_dataset;

        for (auto const &[name, ptr] : algos)
        {
            if (target == "all" || target == to_lower(name))
            {
                resultados_dataset[name] = run_experiments(d, ptr, name, base_seed, num_runs);
            }
            delete ptr;
        }

        imprimir_tabla(d, resultados_dataset);
    }

    return 0;
}