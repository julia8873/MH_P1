#include <cassert>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

#include <extra.h>
#include <parproblem.h>
#include "../common/random.hpp"

using namespace std;
using Random = effolkronium::random_static;

struct pairVirtualSolution { int i, j; };

// =======================
// INICIALIZACIÓN K-MEANS++
// =======================
tSolution<int> inicializacionAvanzada(ParProblem &p) {
    int n = p.getSolutionSize();
    int k = p.getK();
    tSolution<int> sol(n, -1);

    vector<int> centroides;
    centroides.reserve(k);
    centroides.push_back(Random::get(0, n - 1));

    vector<double> dist_min(n, numeric_limits<double>::max());

    for (int c = 1; c < k; ++c) {
        for (int i = 0; i < n; ++i) {
            double d = 0.0;
            const auto &a = p.getData()[i];
            const auto &b = p.getData()[centroides.back()];
            for (size_t di = 0; di < a.size(); ++di) {
                double diff = a[di] - b[di];
                d += diff * diff;
            }
            dist_min[i] = min(dist_min[i], sqrt(d));
        }
        int mejor = 0;
        double max_dist = -1;
        for (int i = 0; i < n; ++i)
            if (dist_min[i] > max_dist) { max_dist = dist_min[i]; mejor = i; }
        centroides.push_back(mejor);
    }

    for (int c = 0; c < k; ++c) sol[centroides[c]] = c + 1;

    for (int i = 0; i < n; ++i) {
        if (sol[i] != -1) continue;
        int mejor_cluster = 1;
        double mejor_dist = numeric_limits<double>::max();
        for (int c = 0; c < k; ++c) {
            double d = 0.0;
            const auto &a = p.getData()[i];
            const auto &b = p.getData()[centroides[c]];
            for (size_t di = 0; di < a.size(); ++di) {
                double diff = a[di] - b[di];
                d += diff * diff;
            }
            d = sqrt(d);
            if (d < mejor_dist) { mejor_dist = d; mejor_cluster = c + 1; }
        }
        sol[i] = mejor_cluster;
    }

    // Reparar restricciones
    for (const auto &res : p.getConstraints()) {
        if (res.type == 1) { // ML
            if (sol[res.i] != sol[res.j]) sol[res.j] = sol[res.i];
        } else { // CL
            if (sol[res.i] == sol[res.j]) sol[res.j] = (sol[res.j] % k) + 1;
        }
    }

    // Evitar clusters vacíos
    vector<int> count(k, 0);
    for (int v : sol) count[v - 1]++;
    for (int c = 0; c < k; ++c) {
        if (count[c] == 0) {
            int origen = max_element(count.begin(), count.end()) - count.begin();
            for (int i = 0; i < n; ++i) {
                if (sol[i] == origen + 1) {
                    sol[i] = c + 1;
                    count[origen]--;
                    count[c]++;
                    break;
                }
            }
        }
    }
    return sol;
}

// =======================
// BÚSQUEDA LOCAL (retorna evals usadas)
// =======================
int busquedaLocal(ParProblem &p, tSolution<int> &sol, double &fitness,
                  vector<int> &num_elem, int evals, int maxevals)
{
    int n = p.getSolutionSize();
    int k = p.getK();

    vector<pairVirtualSolution> vecinos;
    vecinos.reserve(n * k);
    for (int i = 0; i < n; ++i)
        for (int j = 1; j <= k; ++j)
            vecinos.push_back({i, j});

    bool mejora = true;
    while (mejora && evals < maxevals) {
        mejora = false;
        Random::shuffle(vecinos);
        for (auto [pos, nuevo_cluster] : vecinos) {
            if (evals >= maxevals) break;
            if (sol[pos] == nuevo_cluster) continue;
            int antiguo = sol[pos];
            if (num_elem[antiguo - 1] <= 1) continue;

            double delta = p.calcular_nuevo_menos_actual(sol, pos, nuevo_cluster, num_elem);
            evals++;

            if (delta < -1e-10) {
                sol[pos] = nuevo_cluster;
                num_elem[antiguo - 1]--;
                num_elem[nuevo_cluster - 1]++;
                fitness += delta;
                mejora = true;
                break;
            }
        }
    }
    return evals;
}

// =======================
// PERTURBACIÓN (doble movimiento aleatorio fuerte)
// =======================
void perturbar(ParProblem &p, tSolution<int> &sol, vector<int> &num_elem, double &fitness,
               int fuerza)
{
    int n = p.getSolutionSize();
    int k = p.getK();

    // Reasignar aleatoriamente 'fuerza' elementos a clusters distintos
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    Random::shuffle(indices);

    int cambiados = 0;
    for (int i = 0; i < n && cambiados < fuerza; ++i) {
        int pos = indices[i];
        int antiguo = sol[pos];
        if (num_elem[antiguo - 1] <= 1) continue;

        // Nuevo cluster distinto al actual
        int nuevo_cluster = Random::get(1, k);
        if (nuevo_cluster == antiguo)
            nuevo_cluster = (nuevo_cluster % k) + 1;

        sol[pos] = nuevo_cluster;
        num_elem[antiguo - 1]--;
        num_elem[nuevo_cluster - 1]++;
        cambiados++;
    }

    // Recalcular fitness desde cero tras perturbación
    fitness = p.fitness(sol);
}

// =======================
// EXTRA: ILS (Iterated Local Search)
// =======================
ResultMH<int> Extra::optimize(Problem<int> &problem, int maxevals) {
    ParProblem &p = dynamic_cast<ParProblem &>(problem);

    int n = p.getSolutionSize();
    int k = p.getK();

    // Fuerza de perturbación: ~10% de n, mínimo 2
    int fuerza_perturbacion = max(2, n / 10);

    // --- Solución inicial ---
    tSolution<int> sol = inicializacionAvanzada(p);
    double fitness = p.fitness(sol);
    int evals = 1;

    vector<int> num_elem(k, 0);
    for (int v : sol) num_elem[v - 1]++;

    // Primera búsqueda local
    evals = busquedaLocal(p, sol, fitness, num_elem, evals, maxevals);

    // Guardar mejor global
    tSolution<int> mejor_sol = sol;
    double mejor_fitness = fitness;

    // --- ILS: perturbar y relanzar mientras queden evaluaciones ---
    while (evals < maxevals) {
        // Clonar solución actual (partimos del mejor global → reinicio desde allí)
        tSolution<int> sol_pert = mejor_sol;
        double fit_pert = mejor_fitness;

        vector<int> num_elem_pert(k, 0);
        for (int v : sol_pert) num_elem_pert[v - 1]++;

        // Perturbación
        perturbar(p, sol_pert, num_elem_pert, fit_pert, fuerza_perturbacion);
        evals++; // el fitness recalculado en perturbar cuenta como 1 eval

        // Búsqueda local desde solución perturbada
        evals = busquedaLocal(p, sol_pert, fit_pert, num_elem_pert, evals, maxevals);

        // Criterio de aceptación: aceptar si mejora (puede cambiarse a SA)
        if (fit_pert < mejor_fitness) {
            mejor_fitness = fit_pert;
            mejor_sol     = sol_pert;
        }
    }

    return ResultMH<int>(mejor_sol, mejor_fitness, evals);
}

