#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <numeric>

#include "parproblem.h"
#include "greedy.h"
#include "randomsearch.h"
#include "localsearch.h" // Incluimos la BL estándar
#include "localsearchnooptimization.h" // Incluimos la versión no optimizada

using namespace std;

struct AlgoStats {
    string nombre;
    double fit_medio = 0, dist_media = 0, inc_medio = 0, tiempo_medio = 0;
    double eval_medias = 0;
    vector<double> historial_fitness;
};

int main(int argc, char *argv[]) {
    long int base_seed = (argc > 1) ? atoi(argv[1]) : 42;

    // Configuración del problema (Dataset Zoo: k=7, n=101) [cite: 838, 839]
    ParProblem problem(7);
    problem.loadData("../data/zoo_set.dat", "../data/zoo_set_const_15.dat");

    // Instanciación de los 4 algoritmos [cite: 786, 924]
    vector<MH<int>*> algoritmos = { 
        new RandomSearch<int>(), 
        new GreedySearch(), 
        new LocalSearch(), 
        new LocalSearchNoOptimization() 
    };
    // Etiquetas para la tabla y nombres de archivos [cite: 929]
    vector<string> nombres = { "Random", "Greedy", "BL", "BL_NoOpt" };

    cout << "\nResultados para dataset Zoo con 15% restricciones [cite: 928]" << endl;
    cout << left << setw(15) << "Algoritmo" << setw(15) << "Fitness" << setw(15) << "Distancia" 
         << setw(18) << "Incumplimiento" << setw(15) << "Evaluaciones" << "Tiempo (s)" << endl;
    cout << string(95, '-') << endl;

    for (size_t i = 0; i < algoritmos.size(); ++i) {
        AlgoStats s;
        s.nombre = nombres[i];

        for (int run = 0; run < 50; ++run) {
            // Sincronización de semillas (50 ejecuciones por algoritmo) [cite: 865, 870, 871]
            Random::seed(base_seed + run); 
            
            auto t_start = chrono::high_resolution_clock::now();
            // Límite de 100,000 evaluaciones [cite: 884, 922]
            ResultMH<int> res = algoritmos[i]->optimize(problem, 100000); 
            auto t_end = chrono::high_resolution_clock::now();

            chrono::duration<double> diff = t_end - t_start;
            
            s.historial_fitness.push_back(res.fitness);
            s.fit_medio += res.fitness;
            s.eval_medias += res.evaluations;
            s.tiempo_medio += diff.count();
            
            // Métricas: Desviación General e Incumplimiento [cite: 876, 880, 882]
            s.dist_media += problem.calculateDeviation(res.solution); 
            double violations = (double)problem.countViolations(res.solution);
            s.inc_medio += violations / (double)problem.getNumRestricciones();
        }

        // Promedios finales de las 50 ejecuciones [cite: 867, 875]
        s.fit_medio /= 50.0; 
        s.dist_media /= 50.0; 
        s.inc_medio /= 50.0;
        s.eval_medias /= 50.0; 
        s.tiempo_medio /= 50.0;

        // Impresión de resultados agregados [cite: 869, 929]
        cout << left << setw(15) << s.nombre 
             << setw(15) << fixed << setprecision(4) << s.fit_medio 
             << setw(15) << s.dist_media 
             << setw(18) << s.inc_medio 
             << setw(15) << (int)s.eval_medias 
             << s.tiempo_medio << endl;

        // Guardado de datos para el script de Python [cite: 868, 932]
        ofstream f("fitness_" + s.nombre + ".csv");
        for(double val : s.historial_fitness) f << val << "\n";
        f.close();
    }

    // Limpieza de memoria
    for(auto a : algoritmos) delete a;

    return 0;
}