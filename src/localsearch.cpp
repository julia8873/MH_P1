#include <cassert>
#include <localsearch.h>
#include <iostream>
#include <parproblem.h>
using namespace std;

// estructura para representar un vecino:
// mover el elemento i al cluster j
struct pairVirtualSolution
{
    int i, j;
};

ResultMH<int> LocalSearch::optimize(Problem<int> &problem, int maxevals)
{
    ParProblem &p = dynamic_cast<ParProblem &>(problem);

    int n = p.getSolutionSize(); // número de elementos
    int k = p.getK();            // número de clusters

    tSolution<int> sol(n, 0);

    // generar índices y barajarlos
    vector<int> indices_totales(n);
    for (int i = 0; i < n; ++i)
        indices_totales[i] = i;
    Random::shuffle(indices_totales);

    // asignar los k primeros elementos a clusters distintos
    for (int c = 0; c < k; ++c)
        sol[indices_totales[c]] = c + 1;

    // el resto de elementos se asignan aleatoriamente
    for (int i = k; i < n; ++i)
        sol[indices_totales[i]] = Random::get(1, k);

    int evals = 1;

    // calcular fitness inicial
    tFitness fitness_mejor_sol = p.fitness(sol);

    // Para las gráficas
    // p.recordFitness(evals, fitness_mejor_sol);

    bool seguir = true;

    // generar todos los vecinos posibles:
    vector<pairVirtualSolution> vecinos;
    for (int i = 0; i < n; ++i)
        for (int j = 1; j <= k; ++j)
            vecinos.push_back({i, j});

    // contar cuántos elementos hay en cada cluster
    vector<int> num_elementos(k, 0);
    for (int i = 0; i < n; ++i)
        num_elementos[sol[i] - 1]++;

    // Generamos vecinos (intercambiamos dos elem)
    // Si el fitness es mejor, esa es nuestra nueva sol
    // Repetir hasta que no haya mejoras o max de eval
    while (seguir && evals < maxevals)
    {

        seguir = false;
        Random::shuffle(vecinos);

        for (auto [pos, valor] : vecinos)
        {
            if (evals >= maxevals)
                break;

            // si el elemento ya está en ese cluster, no tiene sentido
            if (sol[pos] == valor)
                continue;

            int valor_antiguo = sol[pos];

            // evitar dejar un cluster vacío
            if (num_elementos[valor_antiguo - 1] <= 1)
                continue;

            // probar el movimiento (cambio temporal)
            int prev_val = sol[pos];
            sol[pos] = valor;

            // evaluar la nueva solución
            tFitness fitness_actual = p.fitness(sol);
            evals++;

            // si mejora (menor fitness)
            if (fitness_actual < fitness_mejor_sol)
            {
                fitness_mejor_sol = fitness_actual;

                // actualizar contadores de clusters
                num_elementos[valor_antiguo - 1]--;
                num_elementos[valor - 1]++;

                // Para las gráficas
                // p.recordFitness(evals, fitness_mejor_sol);

                seguir = true;
                // El mejor el primero
                // en cuanto encontramos mejora, volvemos a empezar
                break;
            }
            else
            {
                // si no mejora, deshacer el cambio
                sol[pos] = prev_val;
            }
        }
    }

    return ResultMH<int>(sol, fitness_mejor_sol, evals);
}