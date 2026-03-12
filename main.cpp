#include <iostream>
#include <problem.h>
#include <random.hpp>
#include <string>
#include <util.h>
// Real problem class
#include "parproblem.h" // Sustituimos por tu clase

// All all algorithms
#include "greedy.h"
#include "randomsearch.h"

using namespace std;
int main(int argc, char *argv[]) {
  long int seed;

  // Seed the random number generator
  if (argc == 1) {
    // Use a fixed seed for reproducibility
    seed = 42;
  } else {
    // Tomamos la semilla del primer argumento del programa
    seed = atoi(argv[1]);
  }

  // Create the algorithms
  RandomSearch<int> ralg = RandomSearch<int>();
  // GreedySearch rgreedy = GreedySearch();
  
  // Create the specific problem
  // Supongamos k=7 para el dataset de Zoo y cargamos los archivos
  ParProblem rproblem = ParProblem(7); 
  rproblem.loadData("../data/zoo_set.dat", "../data/zoo_set_const_15.dat");
  
  rproblem.setSeed(seed);
  // Solve using evaluations
  // Aquí metemos los algoritmos en el vector para ejecutarlos en bucle
  vector<pair<string, MH<int> *> > algoritmos = {
    make_pair("RandomSearch", &ralg)
    // make_pair("Greedy", &rgreedy) 
  };
  
  Problem<int> *problem = dynamic_cast<Problem<int> *>(&rproblem);

  for (int i = 0; i < algoritmos.size(); i++) {
    Random::seed(seed);
    
    cout << algoritmos[i].first << endl;
    auto mh = algoritmos[i].second;
    
    ResultMH<int> result = mh->optimize(*problem, 1000);
    
    cout << "Best solution: " << result.solution << endl;
    cout << "Best fitness: " << result.fitness << endl;
    cout << "Evaluations: " << result.evaluations << endl;
    cout << "--------------------------" << endl;
  }

  return 0;
}