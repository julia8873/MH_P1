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

bool ParProblem::loadData(const string& dataPath, const string& constPath) {
    ifstream dataFile(dataPath);
    if (!dataFile.is_open()) {
        cerr << "ERROR: No se pudo abrir el archivo de datos en: " << dataPath << endl;
        return false; // Ahora devolvemos falso en lugar de cerrar el programa
    }
    
    data.clear(); // Limpiar por si se llama varias veces
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

    instanceConstraints.assign(size, vector<Constraint>());

    // leer las restricciones
    ifstream constFile(constPath);
    if (!constFile.is_open()) {
        cerr << "ERROR: No se pudo abrir el archivo de restricciones en: " << constPath << endl;
        return false; 
    } 

    constraints.clear();
    int row = 0;
    while (getline(constFile, line)) {
        stringstream ss(line);
        int col = 0;
        while (getline(ss, cell, ',')) {
            int val = stoi(cell);
            if (row < col && val != 0) {
                Constraint cons = {row, col, val};
                constraints.push_back(cons);
                instanceConstraints[row].push_back(cons);
                instanceConstraints[col].push_back(cons);
            }
            col++;
        }
        row++;
    }
    
    calculateLambda();
    return true; 
}
void ParProblem::setSeed(unsigned int s) {
    global_seed_counter = s;
}

// Contar las violaciones si moviesemos el elemento de instaceId al clusterId
int ParProblem::countInstanceViolations(int instanceId, int clusterId, const tSolution<int>& solution) {
    int violations = 0;
    
    // MEJORA: Recorremos solo las restricciones de esta instancia específica
    for (const auto& constraint : instanceConstraints[instanceId]) {
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
double ParProblem::calcular_nuevo_menos_actual(const tSolution<int>& sol, int pos, int nuevo_c, const vector<int>& num_elem) {
    int antiguo_c = sol[pos];
    if (antiguo_c == nuevo_c) return 0.0;

    int violaciones_antes = countInstanceViolations(pos, antiguo_c, sol);
    int violaciones_nuevas = countInstanceViolations(pos, nuevo_c, sol);
    double dif_viol_antes_nuevas = static_cast<double>(violaciones_nuevas - violaciones_antes) * lambda;


    int valor_antes = num_elem[antiguo_c - 1];
    int valor_nuevo = num_elem[nuevo_c - 1];
    int dims = data[0].size();
    // centroides antes de aplicar cambio
    vector<double> mu_a_antes(dims, 0.0), mu_b_antes(dims, 0.0);
    //centroides despues de aplicar cambio
    vector<double> mu_a_despues(dims, 0.0), mu_b_despues(dims, 0.0);

    // calcular centroides antes
    for (int i = 0; i < size; ++i) {
        if (sol[i] == antiguo_c) {
            for (int d = 0; d < dims; ++d) mu_a_antes[d] += data[i][d];
        } else if (sol[i] == nuevo_c) {
            for (int d = 0; d < dims; ++d) mu_b_antes[d] += data[i][d];
        }
    }

    // calcular centroides después
    for (int d = 0; d < dims; ++d) {
        mu_a_antes[d] /= valor_antes;
        mu_b_antes[d] /= valor_nuevo;
        mu_a_despues[d] = (mu_a_antes[d] * valor_antes - data[pos][d]) / (valor_antes - 1);
        mu_b_despues[d] = (mu_b_antes[d] * valor_nuevo + data[pos][d]) / (valor_nuevo + 1);
    }

    // desviaciones intra-cluster
    double desv_a_antes = 0, desv_a_despues = 0, desv_b_antes = 0, desv_b_despues = 0;
    for (int i = 0; i < size; ++i) {
        if (sol[i] == antiguo_c) {
            desv_a_antes += distanceToExplicitCentroid(i, mu_a_antes);
            if (i != pos) desv_b_antes += distanceToExplicitCentroid(i, mu_a_despues);
        } else if (sol[i] == nuevo_c) {
            desv_a_despues += distanceToExplicitCentroid(i, mu_b_antes);
            desv_b_despues += distanceToExplicitCentroid(i, mu_b_despues);
        }
    }

    // Añadimos el elemento a la desviación del nuevo cluster
    desv_b_despues += distanceToExplicitCentroid(pos, mu_b_despues);

    double d_ant_v = desv_a_antes / valor_antes;
    double d_nue_v = desv_a_despues / valor_nuevo;
    double d_ant_n = desv_b_antes / (valor_antes - 1);
    double d_nue_n = desv_b_despues / (valor_nuevo + 1);
    double delta_desviacion = (d_ant_n + d_nue_n - d_ant_v - d_nue_v) / static_cast<double>(k);

    return delta_desviacion + dif_viol_antes_nuevas;

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


int ParProblem::countViolations(const std::vector<int>& solution) {
    int violations = 0;

    for (const auto& res : constraints) {
        // ML: Deben estar en el mismo cluster. Se viola si son distintos.
        if (res.type == 1) {
            if (solution[res.i] != solution[res.j]) {
                violations++;
            }
        }
        // CL: No pueden estar en el mismo cluster. Se viola si son iguales.
        else if (res.type == -1) {
            if (solution[res.i] == solution[res.j]) {
                violations++;
            }
        }
    }
    return violations;
}