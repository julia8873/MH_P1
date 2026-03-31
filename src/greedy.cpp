#include <cassert>
#include <greedy.h>
#include <iostream>
#include <parproblem.h>

using namespace std;

template <class T>
void print_vector(string name, const vector<T> &sol)
{
    cout << name << ": ";

    for (auto elem : sol)
    {
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
ResultMH<int> GreedySearch::optimize(Problem<int> &problem, int maxevals)
{
    ParProblem &p = dynamic_cast<ParProblem &>(problem);
    int n = p.getSolutionSize();
    int k = p.getK();

    tSolution<int> sol(n, 0);
    int evals = 0;

    // Índices y barajar
    vector<int> indices_totales(n);
    for (int i = 0; i < n; ++i)
        indices_totales[i] = i;
    Random::shuffle(indices_totales);

    // Asignar las k semillas iniciales
    vector<vector<double>> centroids(k);
    for (int i = 0; i < k; ++i)
    {
        int id_semilla = indices_totales[i];
        centroids[i] = p.getDataInstance(id_semilla);
        sol[id_semilla] = i + 1;
    }

    // Borramos los k primeros elem ya asignados
    indices_totales.erase(indices_totales.begin(), indices_totales.begin() + k);

    evals++;
    // Para las gráficas
    // p.recordFitness(evals, p.fitness(sol));

    bool seguir = true;
    // Tenemos los centroides iniciales ya calculados.
    // 1. Iteramos por cada elemento que falte y le asignamos el mejor cluster (bucle for indices)
    // 2. Recalculamos los centroides
    // 3. Si se ha cambiado algun elem de cluster, repetir. (while)
    // Si por ej tenemos en 1D: [1,2,7,8]
    // y los clusters son C1 = [1], C2 = [8]
    // Asignamos en el for: C1 = [1,2], C2 = [7,8]
    // Recalculamos centroides: c1 = 1.5, c2 = 7.5
    // Al volver a iterar ningun elem del conjunto indices-indices iniciales = {2,8} se mueve de su cluster
    // -> Termina
    while (seguir && evals < maxevals)
    {
        seguir = false;
        Random::shuffle(indices_totales);

        // Reasignar instancia al mejor cluster
        for (int id : indices_totales)
        {
            int mejor_cluster = -1;
            int min_violaciones = 999999;
            double min_distancia = 1e18;

            for (int c = 1; c <= k; ++c)
            {
                int violaciones = p.countInstanceViolations(id, c, sol);
                if (violaciones < min_violaciones)
                { // Nuevo cluster tendrá menos violaciones
                    min_violaciones = violaciones;
                    mejor_cluster = c;
                }
                else if (violaciones == min_violaciones)
                { // Criterio: si violaciones coincidían, mirar distancia
                    double dist = p.distanceToExplicitCentroid(id, centroids[c - 1]);
                    if (dist < min_distancia)
                    {
                        min_distancia = dist;
                        mejor_cluster = c;
                    }
                }
            }

            // actualizar si el nuevo cluster es mejor
            if (sol[id] != mejor_cluster)
            {
                sol[id] = mejor_cluster;
                seguir = true;
            }
        }

        if (seguir)
        {
            p.updateCentroids(centroids, sol);
            evals++;
            // Para las gráficas
            // p.recordFitness(evals, p.fitness(sol));
        }
    }

    tFitness f_final = p.fitness(sol);
    return ResultMH<int>(sol, f_final, evals);
}
