#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "../common/problem.h"
#include "../common/random.hpp"

// Ejemplo data
/* 
  size = 101 nº de instancias del zoo
  data[0].size() -> número de atributos de cada animal

  std::vector<std::vector<double>> data = {
    {1.0, 0.0, 0.0, 1.0, 0.0, ...}, // Instancia 0 (ej. Hormiguero)
    {1.0, 0.0, 0.0, 1.0, 0.0, ...}, // Instancia 1 (ej. Antílope)
    {0.0, 0.0, 1.0, 0.0, 0.0, ...}  // Instancia 2 (ej. Bajo)
  };

  std::vector<Constraint> constraints = {
    {0, 1,  1}, // Must-Link: Instancia 0 y 1 deben estar juntas
    {0, 2, -1}, // Cannot-Link: Instancia 0 y 2 no pueden estar juntas
    {1, 5,  1}, // Must-Link: Instancia 1 y 5 deben estar juntas
    {2, 3, -1}  // Cannot-Link: Instancia 2 y 3 no pueden estar juntas
  };

  vector de solución:
  S = [1,2,3,1,2,3]

*/

struct Constraint {
    int i, j, type; // type: 1 para ML, -1 para CL
};

class ParProblem : public Problem<int> {
private:
    size_t size;                             // Número de instancias (n)
    int k;                                   // Número de clusters
    double lambda;                           
    std::vector<std::vector<double>> data;
    std::vector<Constraint> constraints;     // Lista de restricciones para eficiencia [cite: 58, 346]

public:

    // ###################### Funciones heredadas ##################################
    ParProblem(int k_clusters) : Problem<int>(), k(k_clusters), size(0), lambda(0) {}
    virtual size_t getSolutionSize() override { return size; }
    virtual std::pair<int, int> getSolutionDomainRange() override {
        // Dominio -> número de clusters {1,...,k}
        return std::make_pair(1, this->k);
    }
    virtual tFitness fitness(const tSolution<int> &solution) override;
    virtual tSolution<int> createSolution() override;
    bool isValid(const tSolution<int> &solution) override;
    void fix(tSolution<int> &solution) override;

    // ###################### Funciones de esta clase ##################################
    void loadData(const std::string& dataPath, const std::string& constPath);

    void setSeed(unsigned int s);
private:
    // ###################### Funciones auxiliares ##################################
    void calculateLambda();
    double calculateDeviation(const tSolution<int>& sol);

};