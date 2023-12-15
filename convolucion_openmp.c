#include <omp.h>
#include <iostream>
#include <vector>

// Función para aplicar la convolución
void convolucion(const std::vector<std::vector<float>>& imagen, 
                 const std::vector<std::vector<float>>& kernel,
                 std::vector<std::vector<float>>& resultado) {
    int altura = imagen.size();
    int ancho = imagen[0].size();
    int k_altura = kernel.size();
    int k_ancho = kernel[0].size();
    int i, j, m, n;

    #pragma omp parallel for private(i, j, m, n) shared(imagen, kernel, resultado)
    for (i = 0; i < altura - k_altura + 1; ++i) {
        for (j = 0; j < ancho - k_ancho + 1; ++j) {
            float suma = 0.0;
            for (m = 0; m < k_altura; ++m) {
                for (n = 0; n < k_ancho; ++n) {
                    suma += imagen[i + m][j + n] * kernel[m][n];
                }
            }
            resultado[i][j] = suma;
        }
    }
}

int main() {
    // Ejemplo de uso
    std::vector<std::vector<float>> imagen = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<float>> kernel = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    std::vector<std::vector<float>> resultado(1, std::vector<float>(1));

    convolucion(imagen, kernel, resultado);

    // Imprimir el resultado
    for (const auto& fila : resultado) {
        for (float valor : fila) {
            std::cout << valor << ' ';
        }
        std::cout << std::endl;
    }

    return 0;
}

