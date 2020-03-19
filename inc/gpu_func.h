#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector>

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K, cudaStream_t stream=0);

__global__
void gpu_GEMM(const double* __restrict__ dA, const double* __restrict__ dB,
              double* __restrict__ dC, double alpha, double beta,
              int M, int N, int K);

typedef struct raw_params{
    double *W1;
    double *b1;
    double *W2;
    double *b2;
} raw_params;

typedef struct raw_cache{
    double *X;
    double *z1;
    double *a1;
    double *z2;
    double *y;
    double *yhat;
} raw_cache;

typedef struct raw_grad{
    double *dW1;
    double *dW2;
    double *db1;
    double *db2;
} raw_grad;

typedef struct raw_bp{
    double *ydiff;
    double *a1t;
    double *W2t;
    double *Xt;
    double *dz1;
} raw_bp;

void allocate_device_memory(raw_params &d_params, raw_cache &d_cache, raw_grad &d_grad,
                            raw_bp &d_bp, const std::vector<int>& H, int batch_size_node);

// void send_data_to_device(const double *X_local, 
//                          const double *y_local, 
//                          raw_cache &d_cache, 
//                          int batch_size_node, 
//                          int input_dim, 
//                          int output_dim,
//                          cudaStream_t mystream[]);

void send_data_to_device(const double *src, 
    double *dest, 
    int batch_size_node, 
    int dim,
    cudaStream_t stream);

void forward_pass(raw_params &d_params, raw_cache &d_cache, 
                  int input_dim, int h1, int output_dim, int size, cudaStream_t mystream[]);

void backward_pass(raw_params &d_params, raw_cache &d_cache,
                   raw_grad &d_grad, raw_bp &d_bp, int input_dim, int h1, 
                   int output_dim, double reg, int size, int batch_size, 
                   int num_procs,cudaStream_t mystream[]);


void free_all_CUDA(raw_params &d_params, raw_cache &d_cache, raw_grad &d_grad);

void gradient_descent(raw_grad &d_grad, 
                      raw_params &d_params, 
                      double learning_rate,
                      int input_dim,
                      int h1,
                      int output_dim,
                      cudaStream_t mystream[]);

#endif
