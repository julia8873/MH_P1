#include "../inc/parproblem.h"
#include "../common/random.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;

using Random = effolkronium::random_static;

static unsigned int global_seed_counter = 1; // Semilla inicial
const unsigned int SEED_INCREMENT = 10;

// ###################### Funciones heredadas ##################################

// fitness(sol) = Desviacion(sol) + (infeasibility x lambda)
tFitness ParProblem::fitness(const tSolution<int> &solution){
    double desv = calculateDeviation(solution);
    // contar el numero de restricciones violadas
    int infeasibility = 0;
    for (const auto& res : constraints) {  // Recorremos cada restricción
        // constraints: [i,j,type]
        if (res.type == 1) { // Must-Link
            // Miramos el cluster del elemento i y el del elemento j (deberían coincidir)
            if (solution[res.i] != solution[res.j]) infeasibility++;
        } else { // Cannot-Link
            if (solution[res.i] == solution[res.j]) infeasibility++;
        }
    }
    
    return desv + (static_cast<double>(infeasibility) * lambda);
}

tSolution<int> ParProblem::createSolution() {
    Random::seed(global_seed_counter);
    
    global_seed_counter += SEED_INCREMENT;

    tSolution<int> sol(size);
    
    // Creamos un vector de índices [0, 1, 2, ..., size-1]
    // cada numero es un animal pe
    vector<int> indices(size);
    for(int i = 0; i < size; ++i) indices[i] = i;

    // Barajamos los índices para asignar aleatoriamente los k primeros a cada cluster
    // Sino habría que ver otra manera de asegurarse de que ningún cluster esté vacío
    // Podríamos generar k números aleatorios del 0 al size-1 y quitar esos elementos de la lista
    // Pero eso tarda más porque habría que quitar elementos
    // eg: indices = [12,34,56,78,...]
    Random::shuffle(indices);

    // Restricción fuerte es que haya al menos un elemento por cluster
    // Asignamos las primeras k instancias barajadas a los k clusters disponibles
    for (int i = 0; i < k; ++i) {
        sol[indices[i]] = i + 1; // Clusters del 1 al k
    }

    for (size_t i = k; i < size; ++i) {
        // Asignamos al elemento i su cluster k
        // eg: k = 3, indices = [12,34,56,78,98...]
        // en el for anterior: sol = [1,2,3]
        // en este: sol = [1,2,3,(num random del 1-3),...]
        sol[indices[i]] = Random::get(1, k);
    }

    return sol;
}

// La única restricción fuerte en esta practica es que no haya ninguna clase con ningún vector
bool ParProblem::isValid(const tSolution<int> &solution) {
    vector<int> conteo(k, 0);
    for (int cluster_id : solution) conteo[cluster_id - 1]++;
    for (int n : conteo)
        if (n == 0) return false;
    return true;
}

void ParProblem::fix(tSolution<int>& sol) {
    // Si la solución no es válida...
    if (!isValid(sol)) {
        // Opción rápida: la sustituimos por una aleatoria que sabemos que es válida.
        // Esto no es lo más eficiente para buscar, pero evita que el programa falle.
        sol = createSolution(); 
    }
}

// ###################### Funciones de esta clase ##################################

void ParProblem::loadData(const string& dataPath, const string& constPath) {
    ifstream dataFile(dataPath);
    if (!dataFile.is_open()) {
        cerr << "ERROR: No se pudo abrir el archivo de datos en: " << dataPath << endl;
        exit(1); 
    }
    
    string line, cell;

    // leer el .dat
    while (getline(dataFile, line)) {
        vector<double> instance;
        stringstream ss(line);
        while (getline(ss, cell, ',')) {
            instance.push_back(stod(cell));
        }
        data.push_back(instance);
    }
    size = data.size();

    // leer las restricciones
    ifstream constFile(constPath);
    int row = 0;
    while (getline(constFile, line)) {
        stringstream ss(line);
        int col = 0;
        while (getline(ss, cell, ',')) {
            int val = stoi(cell);
            // Guardamos la parte triangular superior de la matriz
            if (row < col && val != 0) {
                constraints.push_back({row, col, val});
            }
            col++;
        }
        row++;
    }
    
    calculateLambda();
}

void ParProblem::setSeed(unsigned int s) {
    global_seed_counter = s;
}

// Contar las violaciones si moviesemos el elemento de instaceId al clusterId
int ParProblem::countInstanceViolations(int instanceId, int clusterId, const tSolution<int>& solution) {
    int violations = 0;
    
    for (const auto& constraint : constraints) {
        // Buscamos restricciones que afecten a la instancia actual
        if (constraint.i == instanceId || constraint.j == instanceId) {
            int otherId = (constraint.i == instanceId) ? constraint.j : constraint.i;
            
            // Solo evaluamos si el otro elemento ya tiene un grupo asignado
            if (solution[otherId] != 0) {
                if (constraint.type == 1) { // Must-Link
                    // Si deben estar juntos pero el otro está en un cluster distinto
                    if (clusterId != solution[otherId]) {
                        violations++;
                    }
                } else { // Cannot-Link
                    // Si no pueden estar juntos pero el otro está en el mismo cluster
                    if (clusterId == solution[otherId]) {
                        violations++;
                    }
                }
            }
        }
    }
    
    return violations;
}

double ParProblem::distanceToExplicitCentroid(int instanceId, const vector<double>& centroid) {
    double sum = 0.0;
    const auto& instanceData = data[instanceId];
    
    // Calculamos la suma de los cuadrados de las diferencias
    for (size_t a = 0; a < instanceData.size(); ++a) {
        double diff = instanceData[a] - centroid[a];
        sum += diff * diff;
    }
    
    return sqrt(sum);
}

void ParProblem::updateCentroids(vector<vector<double>>& centroids, const tSolution<int>& solution) {
    size_t n_atrib = data[0].size();

    // Reiniciar centroides
    centroids.assign(k, vector<double>(n_atrib, 0.0));

    // Contador de elementos por cluster
    vector<int> count(k, 0);

    // 1. Acumular sumas
    for (size_t i = 0; i < solution.size(); ++i) {
        int cluster = solution[i] - 1;
        count[cluster]++;

        const auto& instance = data[i];
        for (size_t j = 0; j < n_atrib; ++j) {
            centroids[cluster][j] += instance[j];
        }
    }

    // 2. Dividir para obtener la media
    for (int c = 0; c < k; ++c) {
        if (count[c] > 0) {
            for (size_t j = 0; j < n_atrib; ++j) {
                centroids[c][j] /= count[c];
            }
        }
    }
}


// si tenemos la solución virtual (a,b), calculamos la desv del cluster sol[a] y la del cluster b
// luego tendremos que calcular las de los mismos clusters pero aplicados el cambio
double ParProblem::calcular_nuevo_menos_actual(const tSolution<int>& sol, int pos, int valor, const vector<int>& num_elem) {
    int old_c_id = sol[pos]; // El cluster actual del elemento 'a'
    int new_c_id = valor;    // El nuevo cluster 'b'

    if (old_c_id == new_c_id) return 0.0;

    // 1. Diferencia en restricciones (infeasibility)
    int vios_viejas = countInstanceViolations(pos, old_c_id, sol);
    int vios_nuevas = countInstanceViolations(pos, new_c_id, sol);
    double delta_vios = static_cast<double>(vios_nuevas - vios_viejas) * lambda;

    // 2. Clasificación de índices para los clusters afectados
    vector<int> idx_old_v, idx_new_v, idx_old_n, idx_new_n;
    
    for (int i = 0; i < size; ++i) {
        if (sol[i] == old_c_id) {
            idx_old_v.push_back(i);
            if (i != pos) idx_old_n.push_back(i); // El 'viejo' cluster tras el cambio
        } else if (sol[i] == new_c_id) {
            idx_new_v.push_back(i);
            idx_new_n.push_back(i); // El 'nuevo' cluster (irá recibiendo el elemento)
        }
    }
    idx_new_n.push_back(pos); // Añadimos el elemento al cluster de destino

    // 3. Cálculo de estados actuales (Antes del cambio)
    vector<double> centroid_old_v = calculateCentroid(idx_old_v);
    vector<double> centroid_new_v = calculateCentroid(idx_new_v);
    double dev_old_v = calculateClusterDeviation(idx_old_v, centroid_old_v);
    double dev_new_v = calculateClusterDeviation(idx_new_v, centroid_new_v);

    // 4. Cálculo de estados virtuales (Después del cambio)
    vector<double> centroid_old_n = calculateCentroid(idx_old_n);
    vector<double> centroid_new_n = calculateCentroid(idx_new_n);
    double dev_old_n = calculateClusterDeviation(idx_old_n, centroid_old_n);
    double dev_new_n = calculateClusterDeviation(idx_new_n, centroid_new_n);

    // 5. Cálculo del incremento de la desviación total
    // La desviación total del problema es la media de las desviaciones de cada cluster:
    // Desviacion = (1/k) * sum(dist_intra_i)
    double delta_desviacion = (dev_old_n + dev_new_n - dev_old_v - dev_new_v) / static_cast<double>(k);

    return delta_desviacion + delta_vios;
}

// ###################### Funciones auxiliares ##################################

void ParProblem::calculateLambda(){
    // lambda de los datos (maxima distancia en el conjunto de datos / nº restricciones del problema)

    // máxima distancia:
    double max_dist = 0;

    for(size_t i = 0; i<size; ++i){
        for(size_t j=i+1; j<size; ++j){ // el vector es simétrico, dist(i,j) = dist(j,i)
        double dist = 0;
        // Distancia entre cada vector (vector i y j)
        // Se podría hacer la distancia euclídea pero como la raíz es una función estrictamente positiva
        // calcularemos la suma de los cuadrados y luego al final le haremos la raíz cuadrada a max_dist
        for(size_t l=0; l<data[i].size(); ++l){
            double dif = data[i][l] - data[j][l];
            dist += dif*dif;
        }

        if(dist > max_dist)
            max_dist = dist;
        }
    }
    max_dist = sqrt(max_dist);

    if (!constraints.empty()) {
        this->lambda = max_dist / static_cast<double>(constraints.size());
    } else {
        this->lambda = 0;
    }
}

// 1. Obtenemos el vector de centroides [mu1,...,muk]
// 2. Calculamos las distancias_intraclusters
// 3. Y la desviacion
// NOTA: se asume que se ha llamado a isValid antes para evitar divisiones por 0
double ParProblem::calculateDeviation(const tSolution<int>& sol){
    // 1. Calculamos los centroides mu_i
        // Creamos un vector de vectores para guardar los índices de cada cluster
        // Por ejemplo: S = [1,2,3,1,2,3]
        // indicesPorCluster sería: [[0,3], [1,4], [2,5]]
    vector<vector<size_t>> indicesPorCluster(k);

    for (size_t i = 0; i < sol.size(); ++i) {
        int cluster_id = sol[i]; 
        indicesPorCluster[cluster_id - 1].push_back(i);
    }

    size_t n_atrib = data[0].size(); 

    // Vector de centroides [mu_1, mu_2, ..., mu_k]
    vector<vector<double>> centroides(k, vector<double>(n_atrib, 0.0));

    updateCentroids(centroides, sol);
    
    // 2. Calcular la distancia intracluster
    // (1/|C_i|)*sum(||x_j-mu_i||)
    vector<double> dist_intra_cluster(k);

    for(size_t i =0; i<k; ++i){
        double dist_euclidea = 0;
        for (size_t idx : indicesPorCluster[i]) {
            double suma = 0;
            for (size_t j = 0; j< n_atrib; ++j) {
                double diferencia = centroides[i][j]-data[idx][j];
                suma += diferencia*diferencia;
            }
            dist_euclidea += sqrt(suma);
        }
        dist_intra_cluster[i] = dist_euclidea/indicesPorCluster[i].size();
    }

    // 3. La desviación:
    // 1/k * sum(distancia intracluster)
    double desviacion = 0;
    for(size_t i=0; i<dist_intra_cluster.size(); ++i){
        desviacion += dist_intra_cluster[i];
    }
    desviacion /= k;

    return desviacion;
}


vector<double> ParProblem::calculateCentroid(const vector<int>& indices) {
    size_t n_atrib = data[0].size();
    vector<double> centroid(n_atrib, 0.0);

    if (indices.empty()) return centroid;

    for (int idx : indices) {
        for (size_t j = 0; j < n_atrib; ++j) {
            centroid[j] += data[idx][j];
        }
    }

    for (double& val : centroid) val /= indices.size();
    
    return centroid;
}

double ParProblem::calculateClusterDeviation(const vector<int>& indices, const vector<double>& centroid) {
    if (indices.empty()) return 0.0;
    
    double total_dist = 0.0;
    for (int idx : indices) {
        total_dist += distanceToExplicitCentroid(idx, centroid);
    }
    
    return total_dist / indices.size();
}