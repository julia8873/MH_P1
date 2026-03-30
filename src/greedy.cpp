#include <cassert>
#include <greedy.h>
#include <iostream>
#include <parproblem.h>

using namespace std;

template <class T> void print_vector(string name, const vector<T> &sol) {
  cout << name << ": ";

  for (auto elem : sol) {
    cout << elem << ", ";
  }
  cout << endl;
}

/**
 *
 * @param problem The problem to be optimized
 * @param maxevals Maximum number of evaluations allowed
 * @return A pair containing the best solution found and its fitness
 */
ResultMH<int> GreedySearch::optimize(Problem<int> &problem, int maxevals) {
    ParProblem &p = dynamic_cast<ParProblem &>(problem);
    int n = p.getSolutionSize();
    int k = p.getK();
    
    tSolution<int> sol(n, 0);

    // cogemos k elementos de data como centroides iniciales
    vector<int> indices_totales(n);
    for(int i = 0; i < n; ++i) indices_totales[i] = i;
    Random::shuffle(indices_totales);  // para que sean aleatorios los centroides
    // asignar los k primeros centroides:
    vector<vector<double>> centroids(k);
    for (int i = 0; i < k; ++i) {
        int id_semilla = indices_totales[i];
        centroids[i] = p.getDataInstance(id_semilla);
        sol[id_semilla] = i + 1;
    }

    // borrar esos elementos de la lista de elementos posibles
    indices_totales.erase(indices_totales.begin(), indices_totales.begin() + k);

    bool seguir = true;
    int evals = 0;
    while (seguir && evals < maxevals) {
        seguir = false;
        
        Random::shuffle(indices_totales);

        // asignar cada instancia al grupo más cercano con menos violaciones
        for (int id : indices_totales) {
            int mejor_cluster = -1;
            int min_violaciones = 999999;
            double min_distancia = 1e18;
            // ver en q grupo meter el elemento
            for (int c = 1; c <= k; ++c) {
                int violaciones = p.countInstanceViolations(id, c, sol);

                if(violaciones < min_violaciones){
                  min_violaciones = violaciones;
                  mejor_cluster = c;
                }else if(violaciones == min_violaciones){
                  double dist = p.distanceToExplicitCentroid(id, centroids[c-1]);
                  if(dist < min_distancia){
                    min_distancia = dist;
                    min_violaciones = violaciones;
                    mejor_cluster = c;
                  }
                }
            }

            if (sol[id] != mejor_cluster) {
                sol[id] = mejor_cluster;
                seguir = true;
            }
        }

        // actualizar centroides con el promedio de sus instancias
        if (seguir) {
            p.updateCentroids(centroids, sol);
        }
        evals++;
    }

    return ResultMH<int>(sol, p.fitness(sol), evals);
}
