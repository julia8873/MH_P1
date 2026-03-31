#include <cassert>
#include <localsearchoptimizado.h>
#include <iostream>
#include <parproblem.h>

using namespace std;

// estructura para representar un vecino:
// mover el elemento i al cluster j
struct pairVirtualSolution
{
    int i, j;
};

/**
 *
 * @param problem The problem to be optimized
 * @param maxevals Maximum number of evaluations allowed
 * @return A pair containing the best solution found and its fitness
 */
ResultMH<int> LocalSearchOptimizado::optimize(Problem<int> &problem, int maxevals)
{
    ParProblem &p = dynamic_cast<ParProblem &>(problem);

    int n = p.getSolutionSize(); // número de elementos
    int k = p.getK();            // número de clusters

    tSolution<int> sol(n, 0);

    // generar índices y barajarlos aleatoriamente
    vector<int> indices_totales(n);
    for (int i = 0; i < n; ++i)
        indices_totales[i] = i;
    Random::shuffle(indices_totales);

    // asignar los k primeros elementos a clusters distintos
    for (int c = 0; c < k; ++c)
        sol[indices_totales[c]] = c + 1;

    // el resto se asigna aleatoriamente
    for (int i = k; i < n; ++i)
        sol[indices_totales[i]] = Random::get(1, k);

    int evals = 1;

    // calcular fitness inicial
    tFitness fitness_mejor_sol = p.fitness(sol);

    // Para las gráficas
    // p.recordFitness(evals, fitness_mejor_sol);

    bool seguir = true;

    // generar todos los vecinos posibles
    vector<pairVirtualSolution> vecinos;
    for (int i = 0; i < n; ++i)
        for (int j = 1; j <= k; ++j)
            vecinos.push_back({i, j});

    // contar elementos en cada cluster
    vector<int> num_elementos(k, 0);
    for (size_t i = 0; i < sol.size(); ++i)
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

            if (sol[pos] == valor)
                continue;

            int valor_antiguo = sol[pos];

            // evitar dejar un cluster vacío
            if (num_elementos[valor_antiguo - 1] <= 1)
                continue;

            // calcular delta de fitness (nuevo - actual)
            // en vez de recalcular todo el fitness
            double nuevo_menos_actual = p.calcular_nuevo_menos_actual(sol, pos, valor, num_elementos);
            evals++;

            // mejora -> el cambio es negatico
            if (nuevo_menos_actual < 0)
            {
                sol[pos] = valor;

                num_elementos[valor_antiguo - 1]--;
                num_elementos[valor - 1]++;

                // actualizar fitness acumulado
                fitness_mejor_sol += nuevo_menos_actual;

                // Para las gráficas
                // p.recordFitness(evals, fitness_mejor_sol);
                seguir = true;
                break;
            }
        }
    }

    return ResultMH<int>(sol, fitness_mejor_sol, evals);
}