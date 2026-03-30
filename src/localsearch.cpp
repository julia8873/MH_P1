#include <cassert>
#include <localsearch.h>
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

struct pairVirtualSolution{
  int i,j;   // (1,2) -> cambiamos el 1-elemento por un 2
};

/**
 *
 * @param problem The problem to be optimized
 * @param maxevals Maximum number of evaluations allowed
 * @return A pair containing the best solution found and its fitness
 */
ResultMH<int> LocalSearch::optimize(Problem<int> &problem, int maxevals) {
    ParProblem &p = dynamic_cast<ParProblem &>(problem);
    int n = p.getSolutionSize();
    int k = p.getK();
    
    tSolution<int> sol(n, 0);
    vector<int> indices_totales(n);
    for (int i = 0; i < n; ++i) indices_totales[i] = i;

    Random::shuffle(indices_totales);

    // Inicialización: aseguramos que cada clúster tenga al menos un elemento [cite: 358, 693]
    for (int c = 0; c < k; ++c) {
        sol[indices_totales[c]] = c + 1;
    }
    for (int i = k; i < n; ++i) {
        sol[indices_totales[i]] = Random::get(1, k);
    }

    int evals = 0;
    tFitness fitness_mejor_sol = p.fitness(sol);
    evals++; // Primera evaluación [cite: 883]

    bool seguir = true;

    // Generar vecindario virtual [cite: 734, 748]
    vector<pairVirtualSolution> vecinos;
    for(int i = 0; i < n; ++i) {
        for(int j = 1; j <= k; ++j) {
            vecinos.push_back({i, j});
        }
    }

    // Contador de elementos por clúster (clústeres 1 a k mapeados a 0 a k-1)
    vector<int> num_elementos(k, 0);
    for (int i = 0; i < n; ++i) {
        num_elementos[sol[i] - 1]++;
    } 

    while(seguir && evals < maxevals){
        seguir = false;
        Random::shuffle(vecinos); // Orden aleatorio para diversidad [cite: 689, 739]

        for(auto [pos, valor] : vecinos){
            if(evals >= maxevals) break;
            
            if (sol[pos] == valor) continue;

            int valor_antiguo = sol[pos]; 

            // RESTRICCIÓN FUERTE: No dejar un clúster vacío [cite: 358, 699, 919]
            if(num_elementos[valor_antiguo - 1] <= 1) continue;

            // Movimiento temporal para evaluar
            int prev_val = sol[pos];
            sol[pos] = valor;
            
            tFitness fitness_actual = p.fitness(sol);
            evals++; // Incremento por cada llamada a fitness()

            if(fitness_actual < fitness_mejor_sol){
                fitness_mejor_sol = fitness_actual;
                seguir = true;
                
                // Actualizamos el contador de elementos de forma definitiva
                num_elementos[valor_antiguo - 1]--;
                num_elementos[valor - 1]++;
                
                // Estrategia "el primer mejor": aceptamos y reiniciamos el vecindario [cite: 687, 688]
                break; 
            } else {
                // Deshacemos el cambio si no hay mejora
                sol[pos] = prev_val;
            }
        }
    }

    return ResultMH<int>(sol, fitness_mejor_sol, evals);
}
