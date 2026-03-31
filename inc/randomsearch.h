#pragma once
#include <mh.h>

using namespace std;

/**
 * Implementation of the Random Search metaheuristic
 *  - Randomly generates solutions and selects the best one
 *
 * @see MH
 * @see Problem
 */
template <typename tDomain> class RandomSearch : public MH<tDomain> {

public:
  RandomSearch() : MH<tDomain>() {}
  virtual ~RandomSearch() {}

  /**
   * Create random solutions until maxevals has been achieved, and returns the
   * best one.
   *
   * @param problem The problem to be optimized (ParProblem)
   * @param maxevals Maximum number of evaluations allowed
   * @return A pair containing the best solution found and its fitness
   */
  ResultMH<int> optimize(Problem<int> &problem, int maxevals) override {

    assert(maxevals > 0);

    tSolution<tDomain> best;
    tFitness best_fitness = -1;

    ParProblem &p = dynamic_cast<ParProblem &>(problem);

    for (int i = 0; i < maxevals; i++) {

      // generar solución aleatoria
      auto solution = problem.createSolution();

      // evaluar fitness
      tFitness fitness = problem.fitness(solution);

      // si es mejor que la mejor actual (o primera iteración)
      if (fitness < best_fitness || best_fitness < 0) {

        best = solution;
        best_fitness = fitness;

        // Para las gráficas (añadir también recordFitness)
      }
    }

    // devolver mejor solución encontrada
    return ResultMH<int>(best, best_fitness, maxevals);
  }
};