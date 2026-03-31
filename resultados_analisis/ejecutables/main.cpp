#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <map>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <limits>

#include "parproblem.h"
#include "greedy.h"
#include "randomsearch.h"
#include "localsearch.h"
#include "localsearchoptimizado.h"
#include "extra.h"
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

vector<string> split(const string &s, char delimitador)
{
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimitador))
        tokens.push_back(token);
    return tokens;
}
Stats run_experiments(const DatasetInfo &d, MH<int> *algo,
                      string nombre_algo, long int base_seed, int num_runs)
{
    ParProblem problem(d.k);
    if (!problem.loadData(d.file_data, d.file_const))
        return Stats();

    vector<vector<double>> historial_convergencia;
    vector<vector<int>>    historial_evals;
    Stats s;

    double best_fitness = numeric_limits<double>::infinity();
    vector<int> best_solution;

    string res_path  = "../resultados/resultados_"  + d.nombre + "_" + d.tag + ".csv";
    string exec_path = "../resultados/ejecuciones_" + d.nombre + "_" + d.tag + ".csv";

    for (int run = 0; run < num_runs; ++run)
    {
        Random::seed(base_seed + run);
        problem.setSeed(base_seed + run);
        problem.clearFitnessHistory();

        auto t_start = chrono::high_resolution_clock::now();
        ResultMH<int> res = algo->optimize(problem, 100000);
        auto t_end   = chrono::high_resolution_clock::now();

        historial_convergencia.push_back(problem.getFitnessHistory());
        historial_evals.push_back(problem.getEvalHistory());

        double tiempo = chrono::duration<double>(t_end - t_start).count();
        double dist   = problem.calculateDeviation(res.solution);
        double inc    = (double)problem.countViolations(res.solution)
                      / (double)problem.getNumRestricciones();

        {
            ofstream f(res_path, ios::app);
            f << nombre_algo << "," << fixed << setprecision(8) << res.fitness << "\n";
        }
        {
            ofstream f(exec_path, ios::app);
            f << nombre_algo << ","
              << fixed << setprecision(8) << res.fitness << ","
              << dist    << ","
              << inc     << ","
              << res.evaluations << ","
              << tiempo  << "\n";
        }

        if (res.fitness < best_fitness)
        {
            best_fitness  = res.fitness;
            best_solution = res.solution;
        }

        s.fit_medio    += res.fitness;
        s.dist_media   += dist;
        s.inc_medio    += inc;
        s.eval_medias  += res.evaluations;
        s.tiempo_medio += tiempo;
    }

    string conv_path = "../resultados/convergencia_" + d.nombre + "_"
                    + d.tag + "_" + nombre_algo + ".csv";
    ofstream f_conv(conv_path);
    f_conv << "Run,Iter,Fitness\n";

    for (int r = 0; r < num_runs; r++)
    {
        const auto &v_fit = historial_convergencia[r];
        const auto &v_ev  = historial_evals[r];

        size_t puntos = min(v_fit.size(), v_ev.size());
for (size_t i = 0; i < puntos; i++) {
    f_conv << r << "," << v_ev[i] << "," << fixed << setprecision(8) << v_fit[i] << "\n";
}
    }
    f_conv.close();

    s.fit_medio    /= num_runs;
    s.dist_media   /= num_runs;
    s.inc_medio    /= num_runs;
    s.eval_medias  /= num_runs;
    s.tiempo_medio /= num_runs;

    ofstream f_resumen("../resultados/resumen_metricas.csv", ios::app);
    f_resumen << nombre_algo << "," << d.nombre << "_" << d.tag << ","
              << s.fit_medio    << "," << s.dist_media << ","
              << s.inc_medio    << "," << s.tiempo_medio << "\n";
    f_resumen.close();

    string sol_path = "../resultados/solucion_" + d.nombre + "_"
                    + d.tag + "_" + nombre_algo + ".csv";
    ofstream f_sol(sol_path);
    f_sol << "Cluster\n";
    for (int cluster_id : best_solution) f_sol << cluster_id << "\n";
    f_sol.close();

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
         << " | " << left << setw(w_fit)  << "Fitness"
         << " | " << left << setw(w_dist) << "Distancia"
         << " | " << left << setw(w_inc)  << "Incumpl."
         << " | " << left << setw(w_ev)   << "Evals."
         << " | " << left << setw(w_time) << "Tiempo (s)" << " |\n";
    cout << "| -" << string(w_alg - 1, '-')
         << " | -" << string(w_fit  - 2, '-') << "-"
         << " | -" << string(w_dist - 2, '-') << "-"
         << " | -" << string(w_inc  - 2, '-') << "-"
         << " | -" << string(w_ev   - 2, '-') << "-"
         << " | -" << string(w_time - 2, '-') << "- |\n";

    vector<string> orden = {"Greedy", "Random", "BL", "BL_Optimizado", "Extra"};
    for (const string &nombre_algo : orden)
    {
        if (resultados_algos.count(nombre_algo))
        {
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

int main(int argc, char *argv[])
{
    auto print_help = []() {
        cout << "\n======================================================================\n";
        cout << "USO ESTANDAR: ./main <algoritmo> [fichero] [semilla] [ejecuciones]\n";
        cout << "----------------------------------------------------------------------\n";
        cout << "  Opciones <algoritmo> : greedy | bl | bl_optimizado | random | extra | all\n";
        cout << "                         (Puedes combinarlos con comas: greedy,bl,extra)\n";
        cout << "  Opciones [fichero]   : zoo | glass | bupa | all\n";
        cout << "  Ejemplo              : ./main bl,greedy zoo 88 10\n";
        cout << "\nUSO EXPLICITO: ./main <algoritmo> custom <fichero_datos> <fichero_const> <k> <etiqueta> [semilla] [ejec.]\n";
        cout << "----------------------------------------------------------------------\n";
        cout << "  Ejemplo              : ./main bl custom ../data/datos.dat ../data/const.dat 5 test1 88 10\n";
        cout << "======================================================================\n\n";
    };

    if (argc < 2)
    {
        cout << "\n[ERROR] No se han proporcionado argumentos.\n";
        print_help();
        return 1;
    }

    string arg_algoritmos = to_lower(argv[1]);

    if (arg_algoritmos == "help" || arg_algoritmos == "-h" || arg_algoritmos == "--help")
    {
        print_help();
        return 0;
    }

    vector<string> target_algos = split(arg_algoritmos, ',');

    vector<string> algos_validos = {"greedy", "bl", "bl_optimizado", "random", "extra", "all"};
    for (const string &t : target_algos)
    {
        if (find(algos_validos.begin(), algos_validos.end(), t) == algos_validos.end())
        {
            cout << "\n[ERROR] Algoritmo desconocido: '" << t << "'.\n";
            cout << "        Algoritmos disponibles: greedy | bl | bl_optimizado | random | extra | all\n";
            print_help();
            return 1;
        }
    }

    string target_data = (argc >= 3) ? to_lower(argv[2]) : "all";

    if (target_data != "custom")
    {
        vector<string> datasets_validos = {"zoo", "glass", "bupa", "all"};
        if (find(datasets_validos.begin(), datasets_validos.end(), target_data) == datasets_validos.end())
        {
            cout << "\n[ERROR] Dataset desconocido: '" << target_data << "'.\n";
            cout << "        Datasets disponibles: zoo | glass | bupa | all\n";
            print_help();
            return 1;
        }
    }

    vector<DatasetInfo> datasets_a_ejecutar;
    long int base_seed = 88;
    int num_runs = 10;

    try
    {
        if (target_data == "custom")
        {
            if (argc < 7)
            {
                cout << "\n[ERROR] Modo 'custom' requiere al menos 6 argumentos.\n";
                print_help();
                return 1;
            }
            datasets_a_ejecutar.push_back({"custom", argv[3], argv[4], stoi(argv[5]), argv[6]});
            base_seed = (argc >= 8) ? stol(argv[7]) : 88;
            num_runs  = (argc >= 9) ? stoi(argv[8]) : 10;
        }
        else
        {
            base_seed = (argc >= 4) ? stol(argv[3]) : 88;
            num_runs  = (argc >= 5) ? stoi(argv[4]) : 10;

            if (num_runs <= 0)
            {
                cout << "\n[ERROR] El número de ejecuciones debe ser un entero positivo.\n";
                return 1;
            }

            vector<DatasetInfo> predefinidos = {
                {"zoo",   "../data/zoo_set.dat",   "../data/zoo_set_const_15.dat",   7,  "15"},
                {"zoo",   "../data/zoo_set.dat",   "../data/zoo_set_const_30.dat",   7,  "30"},
                {"glass", "../data/glass_set.dat", "../data/glass_set_const_15.dat", 7,  "15"},
                {"glass", "../data/glass_set.dat", "../data/glass_set_const_30.dat", 7,  "30"},
                {"bupa",  "../data/bupa_set.dat",  "../data/bupa_set_const_15.dat",  16, "15"},
                {"bupa",  "../data/bupa_set.dat",  "../data/bupa_set_const_30.dat",  16, "30"}
            };
            for (const auto &d : predefinidos)
                if (target_data == "all" ||
                    to_lower(d.nombre).find(target_data) != string::npos)
                    datasets_a_ejecutar.push_back(d);
        }
    }
    catch (const invalid_argument &)
    {
        cout << "\n[ERROR] Argumento no numérico donde se esperaba un número.\n";
        print_help();
        return 1;
    }
    catch (const out_of_range &)
    {
        cout << "\n[ERROR] Valor numérico fuera de rango.\n";
        print_help();
        return 1;
    }
    catch (...)
    {
        cout << "\n[ERROR] Error inesperado al procesar los argumentos.\n";
        print_help();
        return 1;
    }

    if (datasets_a_ejecutar.empty())
    {
        cout << "\n[ERROR] No se encontraron datasets que coincidan con '" << target_data << "'.\n";
        return 0;
    }

    cout << "Iniciando experimentos con semilla base: " << base_seed
         << " y " << num_runs << " ejecuciones por test.\n";

    ofstream f_resumen_init("../resultados/resumen_metricas.csv");
    f_resumen_init << "Algoritmo,Dataset,Fitness,Distancia,Incumplimientos,Tiempo\n";
    f_resumen_init.close();

    for (const auto &d : datasets_a_ejecutar)
    {
        {
            ofstream f("../resultados/resultados_"  + d.nombre + "_" + d.tag + ".csv");
            f << "alg,fitness\n";
        }
        {
            ofstream f("../resultados/ejecuciones_" + d.nombre + "_" + d.tag + ".csv");
            f << "Algoritmo,Fitness,Distancia,Incumplimientos,Evaluaciones,Tiempo\n";
        }

        map<string, MH<int> *> algos;
        algos["Random"]        = new RandomSearch<int>();
        algos["Greedy"]        = new GreedySearch();
        algos["BL"]            = new LocalSearch();
        algos["BL_Optimizado"] = new LocalSearchOptimizado();
        algos["Extra"]         = new Extra();

        map<string, Stats> resultados_dataset;

        for (auto const &[name, ptr] : algos)
        {
            bool ejecutar_este = false;
            for (const string &t_algo : target_algos)
                if (t_algo == "all" || t_algo == to_lower(name))
                    { ejecutar_este = true; break; }

            if (ejecutar_este)
                resultados_dataset[name] = run_experiments(d, ptr, name, base_seed, num_runs);
        }

        if (!resultados_dataset.empty())
            imprimir_tabla(d, resultados_dataset);

        for (auto const &[name, ptr] : algos) delete ptr;
    }

    cout << "\nProceso finalizado. Todo guardado en '../resultados/'" << endl;
    return 0;
}