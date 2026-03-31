#include <cassert>      // Para comprobaciones en tiempo de ejecución
#include <iostream>     // Entrada/salida estándar
#include <vector>       // Uso de vectores dinámicos
#include <limits>       // Para valores extremos (inf, etc.)
#include <algorithm>    // Funciones como shuffle, max_element
#include <extra.h>      // Definiciones propias del proyecto
#include <parproblem.h> // Definición del problema PAR

using namespace std;

// Estructura para representar un vecino: mover elemento i al cluster j
struct pairVirtualSolution
{
    int i, j;
};

// ---------------------------------------------------------------
//          INICIALIZACIÓN DE SOLUCIÓN CON K-MEANS++
// ---------------------------------------------------------------
tSolution<int> inicializarconKplusplus(ParProblem &p)
{
    int n = p.getSolutionSize(); // número de elementos
    int k = p.getK();            // número de clusters
    tSolution<int> sol(n, -1);

    vector<int> centroides;
    centroides.reserve(k);

    // elegir el primer centroide aleatoriamente
    centroides.push_back(Random::get(0, n - 1));

    // para el vector con distancia mínima de cada punto a un centroide
    vector<double> dist_min(n, numeric_limits<double>::max());

    // selección de centroides
    for (int c = 1; c < k; ++c)
    {
        // Actualizar distancias mínimas respecto a centroides ya elegidos
        for (int i = 0; i < n; ++i)
        {
            double d = 0.0;

            const auto &a = p.getData()[i];                 // punto actual
            const auto &b = p.getData()[centroides.back()]; // último centroide

            // distancia euclídea
            for (size_t di = 0; di < a.size(); ++di)
            {
                double diff = a[di] - b[di];
                d += diff * diff;
            }

            // guardar la mínima distancia conocida
            dist_min[i] = min(dist_min[i], sqrt(d));
        }

        // elegir el punto más alejado como nuevo centroide
        int mejor = 0;
        double max_dist = -1;

        for (int i = 0; i < n; ++i)
            if (dist_min[i] > max_dist)
            {
                max_dist = dist_min[i];
                mejor = i;
            }

        centroides.push_back(mejor);
    }

    // Asignar cada centroide a su propio cluster
    for (int c = 0; c < k; ++c)
        sol[centroides[c]] = c + 1;

    // Asignar el resto de puntos al centroide más cercano
    for (int i = 0; i < n; ++i)
    {

        if (sol[i] != -1)
            continue; // Ya es centroide

        int mejor_cluster = 1;
        double mejor_dist = numeric_limits<double>::max();

        for (int c = 0; c < k; ++c)
        {
            double d = 0.0;
            const auto &a = p.getData()[i];
            const auto &b = p.getData()[centroides[c]];

            // Calcular distancia euclídea
            for (size_t di = 0; di < a.size(); ++di)
            {
                double diff = a[di] - b[di];
                d += diff * diff;
            }

            d = sqrt(d);

            // elegir el cluster más cercano
            if (d < mejor_dist)
            {
                mejor_dist = d;
                mejor_cluster = c + 1;
            }
        }

        sol[i] = mejor_cluster;
    }

    // Si hay restricciones
    for (const auto &res : p.getConstraints())
    {

        if (res.type == 1)
        { // Must-Link (ML)
            // si los elementos i y j no están en el mismo cluster
            if (sol[res.i] != sol[res.j])
                // se fuerza que j pase al mismo cluster que i
                sol[res.j] = sol[res.i];
        }
        else
        { // Cannot-Link (CL)
            if (sol[res.i] == sol[res.j])
                sol[res.j] = (sol[res.j] % k) + 1;
        }
    }

    vector<int> count(k, 0);

    // Contar elementos por cluster
    for (int v : sol)
        count[v - 1]++;

    // Si algún cluster está vacío, asignar un elem del cluster que más elementos tenga (no afectará tanto)
    for (int c = 0; c < k; ++c)
    {

        if (count[c] == 0)
        {

            int origen = max_element(count.begin(), count.end()) - count.begin();

            for (int i = 0; i < n; ++i)
            {
                if (sol[i] == origen + 1)
                {

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

// ---------------------------------------------------------------
//              PERTURBACIÓN DE LA SOLUCIÓN
// ---------------------------------------------------------------
void perturbar(ParProblem &p, tSolution<int> &sol, vector<int> &num_elem, double &fitness, int fuerza)
{
    int n = p.getSolutionSize();
    int k = p.getK();

    // vector de índices y barajarlo aleatoriamente
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    Random::shuffle(indices);

    int cambiados = 0;

    // fuerza -> num de elem a mover
    for (int i = 0; i < n && cambiados < fuerza; ++i)
    {

        int pos = indices[i];
        int antiguo = sol[pos];

        // Para evitar dejar un cluster vacío
        if (num_elem[antiguo - 1] <= 1)
            continue;

        // Elegir nuevo cluster distinto
        int nuevo_cluster = Random::get(1, k);
        if (nuevo_cluster == antiguo)
            nuevo_cluster = (nuevo_cluster % k) + 1;

        sol[pos] = nuevo_cluster;
        num_elem[antiguo - 1]--;
        num_elem[nuevo_cluster - 1]++;
        cambiados++;
    }
    fitness = p.fitness(sol);
}

// ---------------------------------------------------------------
//                BÚSQUEDA LOCAL (Optimizada)
// ---------------------------------------------------------------
int busquedaLocal(ParProblem &p, tSolution<int> &sol, double &fitness, vector<int> &num_elem, int evals, int maxevals)
{
    int n = p.getSolutionSize();
    int k = p.getK();

    // Generar todos los posibles vecinos virtuales
    vector<pairVirtualSolution> vecinos;

    for (int i = 0; i < n; ++i)
        for (int j = 1; j <= k; ++j)
            vecinos.push_back({i, j});

    bool mejora = true;
    while (mejora && evals < maxevals)
    {
        mejora = false;
        // Mezclar vecinos para exploración aleatoria
        Random::shuffle(vecinos);

        for (auto [pos, nuevo_cluster] : vecinos)
        {
            if (evals >= maxevals)
                break;
            // Si no cambia nada, saltar
            if (sol[pos] == nuevo_cluster)
                continue;

            int antiguo = sol[pos];
            // Evitar cluster vacío
            if (num_elem[antiguo - 1] <= 1)
                continue;

            // Calcular mejora incremental
            double delta = p.calcular_nuevo_menos_actual(sol, pos, nuevo_cluster, num_elem);
            evals++;

            // Si mejora
            if (delta < 0)
            {
                sol[pos] = nuevo_cluster;
                num_elem[antiguo - 1]--;
                num_elem[nuevo_cluster - 1]++;
                fitness += delta;

                // Para las gráficas:
                // p.recordFitness(evals, fitness);

                mejora = true;
                break; // El primero el mejor
            }
        }
    }

    return evals;
}

ResultMH<int> Extra::optimize(Problem<int> &problem, int maxevals)
{
    // Convertir problema a PAR
    ParProblem &p = dynamic_cast<ParProblem &>(problem);

    int n = p.getSolutionSize();
    int k = p.getK();

    // Ponemos intensidad a un 10% del tamaño del conjunto
    int fuerza_perturbacion = max(2, n / 10);

    tSolution<int> sol = inicializarconKplusplus(p);
    double fitness = p.fitness(sol);
    int evals = 1;

    // Para graficas
    //p.recordFitness(evals, fitness);

    // Contar elementos por cluster
    vector<int> num_elem(k, 0);
    for (int v : sol)
        num_elem[v - 1]++;

    // Aplicar búsqueda local inicial
    evals = busquedaLocal(p, sol, fitness, num_elem, evals, maxevals);

    // Guardar mejor solución
    tSolution<int> mejor_sol = sol;
    double mejor_fitness = fitness;

    while (evals < maxevals)
    {
        // Copiar mejor solución
        tSolution<int> sol_pert = mejor_sol;
        double fit_pert = mejor_fitness;

        // Recalcular tamaños
        vector<int> num_elem_pert(k, 0);
        for (int v : sol_pert)
            num_elem_pert[v - 1]++;

        // Aplicar perturbación
        perturbar(p, sol_pert, num_elem_pert, fit_pert, fuerza_perturbacion);
        evals++;

        // Para gráficas
        // p.recordFitness(evals, fit_pert);
        evals = busquedaLocal(p, sol_pert, fit_pert, num_elem_pert, evals, maxevals);

        // Si es mejor solución
        if (fit_pert < mejor_fitness)
        {
            mejor_fitness = fit_pert;
            mejor_sol = sol_pert;
        }
    }
    return ResultMH<int>(mejor_sol, mejor_fitness, evals);
}