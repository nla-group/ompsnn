#include <iostream>
#include <snn_omp.h>
#include <vector>
#include <omp.h>

int main() {
    // Set the number of threads globally
    omp_set_num_threads(10); // Use 4 threads for all OpenMP regions

    #ifdef OPENBLAS
    openblas_set_num_threads(1); // Disable BLAS threading to avoid oversubscription
    #endif

    int n = 7;
    int d = 3;
    std::vector<FLOAT> data = {
        1.2f, 2.0f, 3.0f,
        2.0f, 2.4f, 2.0f,
        2.0f, 1.0f, 2.0f,
        2.0f, 3.2f, 1.2f,
        2.0f, 3.1f, 2.0f,
        2.0f, 2.2f, 1.0f,
        2.0f, 2.1f, 1.0f
    };

    SNN_FLOAT snn_index(data.data(), n, d);

    std::cout << "mean:" << std::endl;
    print_data(snn_index.mean, 1, d);

    std::cout << "first principal:" << std::endl;
    print_data(snn_index.first_pc, 1, d);

    FLOAT R = 2.0f;

    std::cout << "Single query:" << std::endl;
    std::vector<FLOAT> new_data_unit = {2.3f, 3.2f, 1.0f};
    std::vector<int> indices = snn_index.query_radius(new_data_unit.data(), R);

    std::cout << "Found " << indices.size() << " indices within distance " << R << ":\n";
    std::cout << "Index: ";
    for (int i = 0; i < indices.size(); i++) {
        std::cout << indices[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Multiple queries:" << std::endl;
    std::vector<FLOAT> new_data = {
        2.3f, 2.5f, 1.0f,
        1.5f, 2.5f, 2.5f,
        2.1f, 1.8f, 1.2f
    };

    int m = 3;
    std::vector<std::vector<int>> all_indices = snn_index.query_radius_batch(new_data.data(), m, R);

    for (int j = 0; j < m; j++) {
        std::cout << "Query " << j << " found " << all_indices[j].size() << " indices within distance " << R << ":\n";
        std::cout << "Index: ";
        for (int idx : all_indices[j]) {
            std::cout << idx << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Optional: Print the number of threads used
    std::cout << "Number of threads used: " << omp_get_max_threads() << std::endl;

    return 0;
}
