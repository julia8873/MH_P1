#pragma once
#include <mh.h>

using namespace std;

/**
 * Implementation of the Local Search metaheuristic
 *
 * @see MH
 * @see Problem
 */
// Instanciamos la plantilla con el tipo que nos interese
using MHInt = MH<int>;
using ProblemInt = Problem<int>;
using ResultMHInt = ResultMH<int>;

class Extra : public MHInt {

public:
  Extra() : MH() {}
  virtual ~Extra() {}
  // Implement the MH interface methods
  /**
   *
   * @param problem The problem to be optimized
   * @param maxevals Maximum number of evaluations allowed
   * @return A pair containing the best solution found and its fitness
   */
  virtual ResultMH<int> optimize(Problem<int> &problem, int maxevals);
};
