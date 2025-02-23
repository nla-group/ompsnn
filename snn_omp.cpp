/*
MIT License

Copyright (c) 2022 Stefan Güttel, Xinye Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>
#include <vector>
#include <cstring>
#include <cblas.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <tuple>
#include <iomanip>
#include <omp.h>

typedef float FLOAT;
typedef double DOUBLE;

#define __DEBUG__

#if defined(__DEBUG__)
template <class T>
void print_data(std::vector<T>& data, int n, int d) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << std::setw(3) << std::setprecision(3) << data[i * d + j] << "   ";
        }
        std::cout << std::endl;
    }
}
#endif

class SNN_FLOAT {
private:
    int n; // Number of samples
    int d; // Number of features
    std::vector<FLOAT> data; // Centered data (n x d, row-major)
    std::vector<std::tuple<FLOAT, int, FLOAT>> sorted_proj_idx;

    void compute_projections_and_norms(std::vector<FLOAT>& projections) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0f, data.data(), d,
                    first_pc.data(), 1, 0.0f, projections.data(), 1);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            FLOAT norm_sq = cblas_sdot(d, &data[i * d], 1, &data[i * d], 1);
            sorted_proj_idx[i] = std::make_tuple(projections[i], i, norm_sq);
        }
    }

public:
    std::vector<FLOAT> mean;
    std::vector<FLOAT> first_pc;

    SNN_FLOAT(FLOAT* input_data, int num_samples, int num_features)
        : n(num_samples), d(num_features), data(n * d), mean(d), first_pc(d),
          sorted_proj_idx(n) {
        memcpy(data.data(), input_data, n * d * sizeof(FLOAT));
        compute_first_pc();
    }

    const FLOAT* get_first_pc() const { return first_pc.data(); }

private:
    void compute_first_pc() {
        std::fill(mean.begin(), mean.end(), 0.0f);

        // Parallel mean computation with temp array
        std::vector<FLOAT> temp_mean(d, 0.0f);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < d; j++) {
            FLOAT sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += data[i * d + j];
            }
            temp_mean[j] = sum / n;
        }
        for (int j = 0; j < d; j++) {
            mean[j] = temp_mean[j];
        }

        // Parallel data centering
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                data[i * d + j] -= mean[j];
            }
        }

        // Sequential RNG for simplicity and accuracy
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<FLOAT> dis(-1.0f, 1.0f);
        for (int i = 0; i < d; i++) {
            first_pc[i] = dis(gen);
        }

        std::vector<FLOAT> temp(n);
        const int max_iter = 100;
        const FLOAT tol = 1e-6f;

        // Sequential power iteration for accuracy
        for (int iter = 0; iter < max_iter; iter++) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0f, data.data(), d,
                        first_pc.data(), 1, 0.0f, temp.data(), 1);
            cblas_sgemv(CblasRowMajor, CblasTrans, n, d, 1.0f, data.data(), d,
                        temp.data(), 1, 0.0f, first_pc.data(), 1);

            FLOAT norm = cblas_snrm2(d, first_pc.data(), 1);
            if (norm < 1e-10f) {
                std::cerr << "Zero norm encountered\n";
                break;
            }
            cblas_sscal(d, 1.0f / norm, first_pc.data(), 1);
        }

        std::vector<FLOAT> projections(n);
        compute_projections_and_norms(projections);

        std::sort(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    }

public:
    std::vector<int> query_radius(const FLOAT* new_data, FLOAT R) const {
        FLOAT R_sq = R * R;
        std::vector<FLOAT> centered(d);
        #pragma omp parallel for if(d > 100) schedule(static)
        for (int j = 0; j < d; j++) {
            centered[j] = new_data[j] - mean[j];
        }
        FLOAT q = cblas_sdot(d, first_pc.data(), 1, centered.data(), 1);
        FLOAT new_norm_sq = cblas_sdot(d, centered.data(), 1, centered.data(), 1);

        std::vector<FLOAT> dot_products(n);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0f, data.data(), d,
                    centered.data(), 1, 0.0f, dot_products.data(), 1);

        auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q - R,
                                         [](const auto& p, FLOAT val) { return std::get<0>(p) < val; });
        auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q + R,
                                         [](FLOAT val, const auto& p) { return val < std::get<0>(p); });

        std::vector<int> indices;
        indices.reserve(upper_it - lower_it);
        for (auto it = lower_it; it != upper_it; ++it) {
            int idx = std::get<1>(*it);
            FLOAT dot_xy = dot_products[idx];
            FLOAT norm_sq = std::get<2>(*it);
            FLOAT dist_sq = norm_sq + new_norm_sq - 2.0f * dot_xy;
            if (dist_sq <= R_sq) indices.push_back(idx);
        }
        return indices;
    }

    std::vector<std::vector<int>> query_radius_batch(const FLOAT* new_data, int m, FLOAT R) const {
        FLOAT R_sq = R * R;

        std::vector<FLOAT> centered(m * d);
        #pragma omp parallel for collapse(2) num_threads(4) schedule(static) // Example with specific threads
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++) {
                centered[i * d + j] = new_data[i * d + j] - mean[j];
            }
        }

        std::vector<FLOAT> q_values(m);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, d, 1.0f, centered.data(), d,
                    first_pc.data(), 1, 0.0f, q_values.data(), 1);

        std::vector<FLOAT> new_norm_sq(m);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            new_norm_sq[i] = cblas_sdot(d, centered.data() + i * d, 1, centered.data() + i * d, 1);
        }

        std::vector<FLOAT> dot_products(n * m);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, 1.0f,
                    data.data(), d, centered.data(), d, 0.0f, dot_products.data(), m);

        std::vector<std::vector<int>> all_indices(m);
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < m; j++) {
            FLOAT q = q_values[j];
            auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                             q - R,
                                             [](const auto& p, FLOAT val) { return std::get<0>(p) < val; });
            auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                             q + R,
                                             [](FLOAT val, const auto& p) { return val < std::get<0>(p); });

            std::vector<int>& indices = all_indices[j];
            indices.reserve(upper_it - lower_it);

            for (auto it = lower_it; it != upper_it; ++it) {
                int idx = std::get<1>(*it);
                FLOAT norm_sq = std::get<2>(*it);
                FLOAT dot_xy = dot_products[idx * m + j];
                FLOAT dist_sq = norm_sq + new_norm_sq[j] - 2.0f * dot_xy;
                if (dist_sq <= R_sq) indices.push_back(idx);
            }
        }
        return all_indices;
    }
};

#ifdef OPENBLAS
extern "C" void openblas_set_num_threads(int);
#endif

#if defined(__DEBUG__)
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
#endif