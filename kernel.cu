#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <omp.h>

#define N 15  // Cambia per 1 milione o 10 milioni

// Costanti host
const float scale_host = 1.0005f;
const float Tx_host = 100.0f;
const float Ty_host = -50.0f;
const float Tz_host = 300.0f;

const float R_host[9] = {
    0.866f, -0.433f,  0.25f,
    0.5f,    0.75f,  -0.433f,
    0.0f,    0.5f,   0.866f
};

// Costanti device
__constant__ float scale = 1.0005f;
__constant__ float Tx = 100.0f;
__constant__ float Ty = -50.0f;
__constant__ float Tz = 300.0f;

__constant__ float R[9] = {
    0.866f, -0.433f,  0.25f,
    0.5f,    0.75f,  -0.433f,
    0.0f,    0.5f,   0.866f
};

__global__ void transformGPU(float* x, float* y, float* z, float* x_p, float* y_p, float* z_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x_p[i] = scale * (R[0] * x[i] + R[1] * y[i] + R[2] * z[i]) + Tx;
        y_p[i] = scale * (R[3] * x[i] + R[4] * y[i] + R[5] * z[i]) + Ty;
        z_p[i] = scale * (R[6] * x[i] + R[7] * y[i] + R[8] * z[i]) + Tz;
    }
}

void transformCPU_parallel(float* x, float* y, float* z, float* x_p, float* y_p, float* z_p) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        x_p[i] = scale_host * (R_host[0] * x[i] + R_host[1] * y[i] + R_host[2] * z[i]) + Tx_host;
        y_p[i] = scale_host * (R_host[3] * x[i] + R_host[4] * y[i] + R_host[5] * z[i]) + Ty_host;
        z_p[i] = scale_host * (R_host[6] * x[i] + R_host[7] * y[i] + R_host[8] * z[i]) + Tz_host;
    }
}

int main() {
    float x[N] = { 4577027.76, 4500468.42, 4488649.48, 4427457.57, 4530092.26,
                  4609646.77, 4633236.71, 4642481.80, 4659521.47, 4670821.34,
                  4611855.60, 4684488.19, 4591391.05, 4747685.29, 4884683.96 };

    float y[N] = { 917648.29, 809968.63, 750093.82, 853754.23, 1005607.52,
                  1109858.03, 1102578.81, 1117776.35, 1263146.4, 1340629.32,
                  1265104.19, 1273453.56, 1120190.62, 1383093.64, 1283099.45 };

    float z[N] = { 4331735.53, 4431508.75, 4453849.34, 4495954.92, 4361162.0,
                  4272977.96, 4228319.66, 4198049.47, 4154320.7, 4117494.83,
                  4206376.25, 4122314.59, 4269079.94, 4014906.72, 3882532.57 };

    float x_p_cpu[N], y_p_cpu[N], z_p_cpu[N];
    float x_p_gpu[N], y_p_gpu[N], z_p_gpu[N];

    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    transformCPU_parallel(x, y, z, x_p_cpu, y_p_cpu, z_p_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // GPU
    float* d_x, * d_y, * d_z, * d_xp, * d_yp, * d_zp;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_xp, N * sizeof(float));
    cudaMalloc(&d_yp, N * sizeof(float));
    cudaMalloc(&d_zp, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N * sizeof(float), cudaMemcpyHostToDevice);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    transformGPU << <1, N >> > (d_x, d_y, d_z, d_xp, d_yp, d_zp);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    cudaMemcpy(x_p_gpu, d_xp, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_p_gpu, d_yp, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(z_p_gpu, d_zp, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Confronto errori
    int err_count = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(x_p_cpu[i] - x_p_gpu[i]) > 1e-4 ||
            fabs(y_p_cpu[i] - y_p_gpu[i]) > 1e-4 ||
            fabs(z_p_cpu[i] - z_p_gpu[i]) > 1e-4) {
            err_count++;
        }
    }

    // Info thread e risultati
    int num_threads_cpu = omp_get_max_threads();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "CPU time (parallel): " << cpu_time.count() << " s\n";
    std::cout << "GPU time: " << gpu_time.count() << " s\n";
    std::cout << "Speedup (CPU/GPU): " << cpu_time.count() / gpu_time.count() << "\n";
    std::cout << "Errori CPU/GPU: " << err_count << "\n";
    std::cout << "CPU threads disponibili (OpenMP): " << num_threads_cpu << "\n";
    std::cout << "GPU max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "GPU max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "GPU number of multiprocessors: " << prop.multiProcessorCount << "\n";

    // Cleanup
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_xp); cudaFree(d_yp); cudaFree(d_zp);

    return 0;
}

