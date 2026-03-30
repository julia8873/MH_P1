#include <cassert>
#include <localsearchoptimizado.h>
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
ResultMH<int> LocalSearchOptimizado::optimize(Problem<int> &problem, int maxevals) {
    ParProblem &p = dynamic_cast<ParProblem &>(problem);
    int n = p.getSolutionSize();
    int k = p.getK();
    
    tSolution<int> sol(n, 0);

    // vector de indices [0..n-1]
    vector<int> indices_totales(n);
    for (int i = 0; i < n; ++i)
        indices_totales[i] = i;

    // barajamos
    Random::shuffle(indices_totales);

    // asignamos k elementos aleatorios un cluster del 1 al k 
    // para asegurarnos de que ninguno esté vacío
    for (int c = 0; c < k; ++c) {
        sol[indices_totales[c]] = c + 1;
    }

    // asignamos el resto aleatoriamente
    for (int i = k; i < n; ++i) {
        sol[indices_totales[i]] = Random::get(1, k);
    }

    int evals = 1;
    // Inicializamos con el fitness real de la solución inicial
    tFitness fitness_mejor_sol = p.fitness(sol);
    bool seguir = true;

    // generar los vecinos
    vector<pairVirtualSolution> vecinos;
    for(int i = 0; i < n; ++i) { // Recorremos instancias
        for(int j = 1; j <= k; ++j) { // Recorremos clusters (1 a k)
            vecinos.push_back({i, j});
        }
    }

    // Contador de elementos por cluster para no dejar ningún cluster sin elementos
    vector<int> num_elementos(k, 0);
    for (size_t i = 0; i < sol.size(); ++i) {
        int cluster = sol[i] - 1;
        num_elementos[cluster]++;
    }

    while(seguir && evals < maxevals){
      seguir = false;

      // aleatorizar los vecinos
      Random::shuffle(vecinos);

      for(auto [pos, valor] : vecinos){
        if(evals >= maxevals) break;
        
        // Si la instancia ya está en ese cluster, no hay cambio
        if (sol[pos] == valor) continue;

        // sol = [1,2,3] y vecino = (1,2)
        // guardamos en valor antiguo el elem en la pos 0: 1
        // actualizamos sol con el 2: sol = [2,2,3]
        int valor_antiguo = sol[pos]; 

        // RESTRICCIÓN: No podemos dejar un clúster vacío
        if(num_elementos[valor_antiguo - 1] <= 1) continue;

        // Definimos nuevo_menos_actual como f(S') - f(S)
        // S la solución actual
        // S' la solución con los vecinos aplicados
        // entonces f(S') = f(S) + nuevo_menos_actual (ver memoria para la explicación)
        double nuevo_menos_actual = p.calcular_nuevo_menos_actual(sol, pos, valor, num_elementos);
        evals++;

        // si el cambio es negativo es que la solución es mejor (minimizamos)
        if(nuevo_menos_actual < -1e-10) {
          sol[pos] = valor;
          
          num_elementos[valor_antiguo - 1]--;
          num_elementos[valor - 1]++;
          
          fitness_mejor_sol += nuevo_menos_actual;
          
          seguir = true; 
          break;
        }
      }
    }

    // Devolvemos el fitness acumulado para evitar una llamada extra a fitness() si no es necesario,
    // o usamos p.fitness(sol) para asegurar precisión total al finalizar.
    return ResultMH<int>(sol, fitness_mejor_sol, evals);
}