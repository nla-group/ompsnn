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

    std::vector<std::tuple<FLOAT, int, FLOAT>> sorted_proj_idx; // (projection, index, squared_norm)

    void compute_projections_and_norms(std::vector<FLOAT>& projections) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d,
                    1.0f, data.data(), d, first_pc.data(), 1,
                    0.0f, projections.data(), 1);
        for (int i = 0; i < n; i++) {
            FLOAT norm_sq = cblas_sdot(d, &data[i * d], 1, &data[i * d], 1);
            sorted_proj_idx[i] = std::make_tuple(projections[i], i, norm_sq);
        }
    }

public:
    std::vector<FLOAT> mean; // Feature means (d)
    std::vector<FLOAT> first_pc; // First principal component (d x 1)

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
        for (int j = 0; j < d; j++) {
            FLOAT sum = 0.0f;
            for (int i = 0; i < n; i++) sum += data[i * d + j];
            mean[j] = sum / n;
            for (int i = 0; i < n; i++) data[i * d + j] -= mean[j];
        }

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<FLOAT> dis(-1.0f, 1.0f);
        for (int i = 0; i < d; i++) first_pc[i] = dis(gen);

        std::vector<FLOAT> temp(n);
        const int max_iter = 100;
        FLOAT norm_prev = 0.0f;

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
            norm_prev = norm;
        }

        std::vector<FLOAT> projections(n);
        compute_projections_and_norms(projections);
        std::sort(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    }

public:
    // Single query method 
    std::vector<int> query_radius(const FLOAT* new_data, FLOAT R) const {
        FLOAT R_sq = R * R;
        FLOAT centered[d];
        for (int j = 0; j < d; j++) centered[j] = new_data[j] - mean[j];
        FLOAT q = cblas_sdot(d, first_pc.data(), 1, centered, 1);
        FLOAT new_norm_sq = cblas_sdot(d, centered, 1, centered, 1);

        std::vector<FLOAT> dot_products(n);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0f, data.data(), d,
                    centered, 1, 0.0f, dot_products.data(), 1);

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
            if (dist_sq <= R_sq) {
                indices.push_back(idx);

            }
        }

        return indices;
    }

    // batch queries method
    std::vector<std::vector<int>> query_radius_batch(const FLOAT* new_data, int m, FLOAT R) const {
        FLOAT R_sq = R * R;

        // Step 1: Center all query points
        std::vector<FLOAT> centered(m * d);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++) {
                centered[i * d + j] = new_data[i * d + j] - mean[j];
            }
        }

        // Step 2: Compute projections (q) for all queries
        std::vector<FLOAT> q_values(m);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, d, 1.0f, centered.data(), d,
                    first_pc.data(), 1, 0.0f, q_values.data(), 1);

        // Step 3: Compute squared norms for all query points
        std::vector<FLOAT> new_norm_sq(m);
        for (int i = 0; i < m; i++) {
            new_norm_sq[i] = cblas_sdot(d, &centered[i * d], 1, &centered[i * d], 1);
        }

        // Step 4: Precompute dot products (n x m matrix)
        std::vector<FLOAT> dot_products(n * m);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, 1.0f,
                    data.data(), d, centered.data(), d, 0.0f, dot_products.data(), m);

        // Step 5: Process each query
        std::vector<std::vector<int>> all_indices(m);
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
                FLOAT dot_xy = dot_products[idx * m + j]; // Access n x m matrix
                FLOAT dist_sq = norm_sq + new_norm_sq[j] - 2.0f * dot_xy;
                if (dist_sq <= R_sq) {
                    indices.push_back(idx);
                }
            }

        }
        return all_indices;
    }
};




class SNN_DOUBLE { // Renamed class to reflect double precision
private:
    int n; // Number of samples
    int d; // Number of features
    std::vector<DOUBLE> data; // Centered data (n x d, row-major)

    std::vector<std::tuple<DOUBLE, int, DOUBLE>> sorted_proj_idx; // (projection, index, squared_norm)

    void compute_projections_and_norms(std::vector<DOUBLE>& projections) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, d,
                    1.0, data.data(), d, first_pc.data(), 1,
                    0.0, projections.data(), 1); // Changed to double-precision
        for (int i = 0; i < n; i++) {
            DOUBLE norm_sq = cblas_ddot(d, &data[i * d], 1, &data[i * d], 1); // Changed to ddot
            sorted_proj_idx[i] = std::make_tuple(projections[i], i, norm_sq);
        }
    }

public:
    std::vector<DOUBLE> mean; // Feature means (d)
    std::vector<DOUBLE> first_pc; // First principal component (d x 1)

    SNN_DOUBLE(DOUBLE* input_data, int num_samples, int num_features)
        : n(num_samples), d(num_features), data(n * d), mean(d), first_pc(d),
          sorted_proj_idx(n) {

        memcpy(data.data(), input_data, n * d * sizeof(DOUBLE));
        compute_first_pc();
    }

    const DOUBLE* get_first_pc() const { return first_pc.data(); }

private:
    void compute_first_pc() {
        std::fill(mean.begin(), mean.end(), 0.0);
        for (int j = 0; j < d; j++) {
            DOUBLE sum = 0.0;
            for (int i = 0; i < n; i++) sum += data[i * d + j];
            mean[j] = sum / n;
            for (int i = 0; i < n; i++) data[i * d + j] -= mean[j];
        }

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<DOUBLE> dis(-1.0, 1.0); // Updated to DOUBLE
        for (int i = 0; i < d; i++) first_pc[i] = dis(gen);

        std::vector<DOUBLE> temp(n);
        const int max_iter = 100;
        DOUBLE norm_prev = 0.0;

        for (int iter = 0; iter < max_iter; iter++) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0, data.data(), d,
                        first_pc.data(), 1, 0.0, temp.data(), 1); // Changed to dgemv
            cblas_dgemv(CblasRowMajor, CblasTrans, n, d, 1.0, data.data(), d,
                        temp.data(), 1, 0.0, first_pc.data(), 1); // Changed to dgemv

            DOUBLE norm = cblas_dnrm2(d, first_pc.data(), 1); // Changed to dnrm2
            if (norm < 1e-10) { // Adjusted threshold
                std::cerr << "Zero norm encountered\n";
                break;
            }
            cblas_dscal(d, 1.0 / norm, first_pc.data(), 1); // Changed to dscal
            norm_prev = norm;
        }

        std::vector<DOUBLE> projections(n);
        compute_projections_and_norms(projections);
        std::sort(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    }

public:
    std::vector<int> query_radius(const DOUBLE* new_data, DOUBLE R) const {
        DOUBLE R_sq = R * R;
        DOUBLE centered[d];
        for (int j = 0; j < d; j++) centered[j] = new_data[j] - mean[j];
        DOUBLE q = cblas_ddot(d, first_pc.data(), 1, centered, 1); // Changed to ddot
        DOUBLE new_norm_sq = cblas_ddot(d, centered, 1, centered, 1); // Changed to ddot

        std::vector<DOUBLE> dot_products(n);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0, data.data(), d,
                    centered, 1, 0.0, dot_products.data(), 1); // Changed to dgemv

        auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q - R,
                                         [](const auto& p, DOUBLE val) { return std::get<0>(p) < val; });
        auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q + R,
                                         [](DOUBLE val, const auto& p) { return val < std::get<0>(p); });

        std::vector<int> indices;
        indices.reserve(upper_it - lower_it);
        for (auto it = lower_it; it != upper_it; ++it) {
            int idx = std::get<1>(*it);
            DOUBLE dot_xy = dot_products[idx];
            DOUBLE norm_sq = std::get<2>(*it);
            DOUBLE dist_sq = norm_sq + new_norm_sq - 2.0 * dot_xy;
            if (dist_sq <= R_sq) {
                indices.push_back(idx);
            }
        }

        return indices;
    }

    std::vector<std::vector<int>> query_radius_batch(const DOUBLE* new_data, int m, DOUBLE R) const {
        DOUBLE R_sq = R * R;

        std::vector<DOUBLE> centered(m * d);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++) {
                centered[i * d + j] = new_data[i * d + j] - mean[j];
            }
        }

        std::vector<DOUBLE> q_values(m);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, d, 1.0, centered.data(), d,
                    first_pc.data(), 1, 0.0, q_values.data(), 1); // Changed to dgemv

        std::vector<DOUBLE> new_norm_sq(m);
        for (int i = 0; i < m; i++) {
            new_norm_sq[i] = cblas_ddot(d, &centered[i * d], 1, &centered[i * d], 1); // Changed to ddot
        }

        std::vector<DOUBLE> dot_products(n * m);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, 1.0,
                    data.data(), d, centered.data(), d, 0.0, dot_products.data(), m); // Changed to dgemm

        std::vector<std::vector<int>> all_indices(m);
        for (int j = 0; j < m; j++) {
            DOUBLE q = q_values[j];
            auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                             q - R,
                                             [](const auto& p, DOUBLE val) { return std::get<0>(p) < val; });
            auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                             q + R,
                                             [](DOUBLE val, const auto& p) { return val < std::get<0>(p); });

            std::vector<int>& indices = all_indices[j];
            indices.reserve(upper_it - lower_it);

            for (auto it = lower_it; it != upper_it; ++it) {
                int idx = std::get<1>(*it);
                DOUBLE norm_sq = std::get<2>(*it);
                DOUBLE dot_xy = dot_products[idx * m + j];
                DOUBLE dist_sq = norm_sq + new_norm_sq[j] - 2.0 * dot_xy;
                if (dist_sq <= R_sq) {
                    indices.push_back(idx);
                }
            }
        }
        return all_indices;
    }
};



#if defined(__DEBUG__)
int main() {
    int n = 7; // Samples
    int d = 3; // Features
    std::vector<FLOAT> data = {
        1.2, 2.0, 3.0,
        2.0, 2.4, 2.0,
        2.0, 1.0, 2.0,
        2.0, 3.2, 1.2,
        2.0, 3.1, 2.0,
        2.0, 2.2, 1.0,
        2.0, 2.1, 1.0
    };

    SNN_FLOAT snn_index(data.data(), n, d);

    std::cout << "mean:" << std::endl;
    print_data(snn_index.mean, 1, d);

    std::cout << "first principal:" << std::endl;
    print_data(snn_index.first_pc, 1, d);


    FLOAT R = 2.0f;

    std::cout << "Single query:" << std::endl;
    // Create a new data point (1 x d)
    std::vector<FLOAT> new_data_unit = {2.3, 3.2, 1.0};
    std::vector<int> indices = snn_index.query_radius(new_data_unit.data(), R);

    // Output results
    std::cout << "Found " << indices.size() << " indices within distance " << R << ":\n";
    std::cout << "Index: ";
    for (int i = 0; i < indices.size(); i++) {
        std::cout << indices[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Mutiple queries:" << std::endl;

    std::vector<FLOAT> new_data = {
        2.3, 3.2, 1.0,  // Query 0
        1.5, 2.5, 2.5,  // Query 1
        2.1, 1.8, 1.2   // Query 2
    };

    // batch query points (m = 3, d = 3)
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
    return 0;
}
#endif