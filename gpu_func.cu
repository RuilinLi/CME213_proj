#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_result));
    return result;
}

#define BLOCK_SIZE 16

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
// int myGEMM(double* __restrict__ A, double* __restrict__ B,
//            double* __restrict__ C, double* alpha, double* beta,
//            int M, int N, int K) {
//     /* TODO: Write an efficient GEMM implementation on GPU */
//     dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
//     int num_block_x = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;
//     int num_block_y = (M + BLOCK_SIZE - 1)/BLOCK_SIZE;
//     dim3 blocks(num_block_x, num_block_y);

//     gpu_GEMM<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K);
//     return 0;
// }

// __global__
// void gpu_GEMM(double* __restrict__ dA, double* __restrict__ dB,
//               double* __restrict__ dC, double alpha, double beta,
//               int M, int N, int K)
// {
// // Note this implementation requires blockDim.y = blockDim.x
// // C is M by N, A is M by K, B is K by N
//     int Cx = blockIdx.x*blockDim.x + threadIdx.x;
//     int Cy = blockIdx.y*blockDim.y + threadIdx.y;

//     double C_val = 0.0;
//     int num_step = (K + BLOCK_SIZE - 1)/BLOCK_SIZE;
//     for(int i = 0; i < num_step; ++i){
//         __shared__ double As[BLOCK_SIZE * BLOCK_SIZE];
//         __shared__ double Bs[BLOCK_SIZE * BLOCK_SIZE];
//         int Ax_global = threadIdx.x + i*BLOCK_SIZE;
//         As[threadIdx.x*BLOCK_SIZE + threadIdx.y] = (Ax_global < K && Cy < M) ? dA[Ax_global*M + Cy]:0.0;
//         int By_global = threadIdx.y + i*BLOCK_SIZE;
//         Bs[threadIdx.x*BLOCK_SIZE + threadIdx.y] = (By_global < K && Cx < N) ? dB[Cx*K + By_global]:0.0;
//         __syncthreads();

//         for (int j = 0; j < BLOCK_SIZE; ++j){
//             C_val += As[threadIdx.y+BLOCK_SIZE*j]*Bs[j+BLOCK_SIZE*threadIdx.x];
//         }

//         __syncthreads();
//     }
//     if (Cx < N && Cy < M){
//         dC[Cx*M+Cy] = alpha*C_val + beta*dC[Cx*M+Cy];
//     }
// }


// Second implementation 
/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
    double* __restrict__ C, double* alpha, double* beta,
    int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    dim3 threads(16, 4);
    int num_block_x = (N + 16 - 1)/16;
    int num_block_y = (M + 64 - 1)/64;
    dim3 blocks(num_block_x, num_block_y);

    gpu_GEMM<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K);
    return 0;
}

// This is indeed faster
__global__
void gpu_GEMM(double* __restrict__ dA, double* __restrict__ dB,
               double* __restrict__ dC, double alpha, double beta,
               int M, int N, int K)
{
    int col = blockIdx.x*16;
    int row = blockIdx.y*64;
    int Cx = blockIdx.x*16 + threadIdx.x;
    int num_step = (K + 4 - 1)/4;
    double a[4];
    int row_offset = threadIdx.x+16*threadIdx.y;
    double C_val[16];
    for(int k = 0; k < 16; ++k){
        C_val[k] = 0.0;
    }
    for (int i  = 0; i < num_step; ++i){
        __shared__ double Bs[4*16];
        Bs[threadIdx.x*4+threadIdx.y] = (Cx < N && 4*i+threadIdx.y<K)?dB[Cx*K+4*i+threadIdx.y]:0.0;
        a[0] = (4*i<K && row+row_offset<M)?dA[(4*i)*M+row+row_offset]:0.0;
        a[1] = (4*i+1<K && row+row_offset<M)?dA[(4*i+1)*M+row+row_offset]:0.0;
        a[2] = (4*i+2<K && row+row_offset<M)?dA[(4*i+2)*M+row+row_offset]:0.0;
        a[3] = (4*i+3<K && row+row_offset<M)?dA[(4*i+3)*M+row+row_offset]:0.0;

        __syncthreads();
        for (int k = 0; k < 16; ++k){
            C_val[k] += a[0]*Bs[k*4] + a[1]*Bs[k*4+1] +a[2]*Bs[k*4+2] +a[3]*Bs[k*4+3];
        }
        __syncthreads();
    }
    for (int k = 0; k < 16; ++k){
        if( col+k < N && row+row_offset <M){
            dC[(col+k)*M+row+row_offset] = alpha*C_val[k] + beta*dC[(col+k)*M+row+row_offset];
        }
    }
}

// This kernel add b to each column of Z in place
// Z is M by N, b has length M
__global__
void gpu_add_col(double *Z, double *b, int M, int N){
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    int indy = blockIdx.y*blockDim.y + threadIdx.y;
    if (indx < N && indy < M){
        Z[indx*M+indy] += b[indy];
    }
}

int add_col(double *Z, double *b, int M, int N)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int num_block_x = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;
    int num_block_y = (M + BLOCK_SIZE - 1)/BLOCK_SIZE;
    dim3 blocks(num_block_x, num_block_y);

    gpu_add_col<<<blocks, threads>>>(Z, b, M, N);
    return 0;
}

// This kernel computes the sigmoid of the first matrix
// and save it to the second
// Z has dimension M by N
__global__
void sigmoid_gpu(double *Z, double* a, int M, int N)
{
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    int indy = blockIdx.y*blockDim.y + threadIdx.y;
    if (indx < N && indy < M){
        a[indx*M+indy] = 1.0/(1+exp(-Z[indx*M+indy]));
    }

}

int sigmoid(double *Z, double* a, int M, int N)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int num_block_x = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;
    int num_block_y = (M + BLOCK_SIZE - 1)/BLOCK_SIZE;
    dim3 blocks(num_block_x, num_block_y);

    sigmoid_gpu<<<blocks, threads>>>(Z, a, M, N);
    return 0;
}

// Function that computes softmax, assuming the output dimension
// is small
__global__
void softmax_gpu(double *Z, double *a, int M, int N){
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    int indy = blockIdx.y*blockDim.y + threadIdx.y;
    if (indx < N && indy < M){
        double denom = 0.0;
        for (int i = 0; i < M; ++i){
            denom += exp(Z[M*indx + i]);
        }
        a[indx*M+indy] = exp(Z[M*indx+indy])/denom;
    }
}

int softmax(double *Z, double *a, int M, int N){
    dim3 threads(64, 2);
    int num_block_x = (N + 64 - 1)/64;
    int num_block_y = (M + 2 - 1)/2;
    dim3 blocks(num_block_x, num_block_y);
    softmax_gpu<<<blocks, threads>>>(Z, a, M, N);
    return 0;
}


// Kernel that compute a * A + b * B and save it to C,
// a, b are scalars. A,B, C are M by N,
__global__
void matadd_gpu(double *A, double *B, double *C, int M, int N, double a, double b)
{    
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    int indy = blockIdx.y*blockDim.y + threadIdx.y;
    if (indx < N && indy < M){
        C[indx*M + indy] = a*A[indx*M + indy] + b*B[indx*M + indy];
    }
}

int matadd(double *A, double *B, double *C, int M, int N, double a, double b)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int num_block_x = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;
    int num_block_y = (M + BLOCK_SIZE - 1)/BLOCK_SIZE;
    dim3 blocks(num_block_x, num_block_y);

    matadd_gpu<<<blocks, threads>>>(A, B, C, M, N, a, b);
    return 0;

}


// This kernel transpose the matrix A and save it to
// At, A is M by N, for now do it naively
__global__
void transpose_gpu(double *A, double *At, int M, int N){
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    int indy = blockIdx.y*blockDim.y + threadIdx.y;
    if(indx < N && indy < M){
        At[indx + indy*N] = A[indy + indx*M];
    }
}

int transpose(double *A, double *At, int M, int N){
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int num_block_x = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;
    int num_block_y = (M + BLOCK_SIZE - 1)/BLOCK_SIZE;
    dim3 blocks(num_block_x, num_block_y);

    transpose_gpu<<<blocks, threads>>>(A, At, M, N);
    return 0;
}

// This kernel sum the rows of the matrix A and store 
// it to the entries of b. A is M by N
__global__
void naive_reduce_sum(double *A, double *b, int M, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < M){
        double result = 0.0;
        for (int j = 0; j < N; ++j){
            result += A[idx+j*M];
        }
        b[idx] = result;
    }
}

int naive_sum(double *A, double *b, int M, int N)
{
    int thread = 1;
    int block = M;
    naive_reduce_sum<<<block, thread>>>(A, b, M, N);
    return 0;
}

// This is a specialized kernel to compute dCE/dz1
// dz1 has dimension M by N
__global__
void get_dz1_gpu(double *dz1, double *a, int M, int N)
{
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    int indy = blockIdx.y*blockDim.y + threadIdx.y;
    if (indx < N && indy < M){
        dz1[indx*M+indy] *= (a[indx*M+indy]*(1-a[indx*M+indy]));
    }
}

int get_dz1(double *dz1, double *a, int M, int N){
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int num_block_x = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;
    int num_block_y = (M + BLOCK_SIZE - 1)/BLOCK_SIZE;
    dim3 blocks(num_block_x, num_block_y);
    get_dz1_gpu<<<blocks, threads>>>(dz1, a, M, N);
    return 0;
}


void allocate_device_memory(raw_params &d_params,
                            raw_cache &d_cache,
                            raw_grad &d_grad,
                            raw_bp &d_bp,
                            const std::vector<int>& H,
                            int batch_size_node)
{
    cudaMalloc((void**)&d_params.W1, sizeof(double) * H[0] * H[1]);
    cudaMalloc((void**)&d_params.b1, sizeof(double) * H[1]);
    cudaMalloc((void**)&d_params.W2, sizeof(double) * H[1] * H[2]);
    cudaMalloc((void**)&d_params.b2, sizeof(double) * H[2]);

    cudaMalloc((void**)&d_cache.X, sizeof(double) * H[0] * batch_size_node);
    cudaMalloc((void**)&d_cache.z1, sizeof(double) * H[1] * batch_size_node);
    cudaMalloc((void**)&d_cache.a1, sizeof(double) * H[1] * batch_size_node);
    cudaMalloc((void**)&d_cache.z2, sizeof(double) * H[2] * batch_size_node);
    cudaMalloc((void**)&d_cache.y, sizeof(double) * H[2] * batch_size_node);
    cudaMalloc((void**)&d_cache.yhat, sizeof(double) * H[2] * batch_size_node);

    cudaMalloc((void**)&d_grad.dW1, sizeof(double) * H[0] * H[1]);
    cudaMalloc((void**)&d_grad.db1, sizeof(double) * H[1]);
    cudaMalloc((void**)&d_grad.dW2, sizeof(double) * H[1] * H[2]);
    cudaMalloc((void**)&d_grad.db2, sizeof(double) * H[2]);

    cudaMalloc((void**)&d_bp.ydiff, sizeof(double) * H[2]*batch_size_node);
    cudaMalloc((void**)&d_bp.a1t, sizeof(double) * H[1] * batch_size_node);
    cudaMalloc((void**)&d_bp.W2t, sizeof(double) * H[1] * H[2]);
    cudaMalloc((void**)&d_bp.Xt, sizeof(double) * H[0] * batch_size_node);
    cudaMalloc((void**)&d_bp.dz1, sizeof(double) * H[1] * batch_size_node);
    

}


void send_data_to_device(const double *X_local, 
                         const double *y_local, 
                         raw_cache &d_cache, 
                         int batch_size_node, 
                         int input_dim, 
                         int output_dim)
{
    cudaMemcpy(d_cache.X, X_local, sizeof(double) * batch_size_node * input_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cache.y, y_local, sizeof(double) * batch_size_node * output_dim, cudaMemcpyHostToDevice);
}


// size is the number of images in this batch
void forward_pass(raw_params &d_params, raw_cache &d_cache, int input_dim, int h1, int output_dim, int size)
{
    double alpha = 1.0;
    double beta = 0.0;
    myGEMM(d_params.W1, d_cache.X, d_cache.z1, &alpha, &beta, h1, size, input_dim);
    add_col(d_cache.z1, d_params.b1, h1, size);
    sigmoid(d_cache.z1, d_cache.a1, h1, size);
    myGEMM(d_params.W2, d_cache.a1, d_cache.z2, &alpha, &beta, output_dim, size, h1);
    add_col(d_cache.z2, d_params.b2, output_dim, size);
    softmax(d_cache.z2, d_cache.yhat, output_dim, size);
}

void backward_pass(raw_params &d_params,
                   raw_cache &d_cache,
                   raw_grad &d_grad, 
                   raw_bp &d_bp,
                   int input_dim, 
                   int h1, 
                   int output_dim,
                   double reg,
                   int size,
                   int batch_size,
                   int num_procs)
{
    double alpha = 1.0;
    double beta = 0.0;
    matadd(d_cache.yhat, d_cache.y, d_bp.ydiff, output_dim, size, 1.0/(double)batch_size, -1.0/(double)batch_size);
    transpose(d_cache.a1, d_bp.a1t, h1, size);
    transpose(d_params.W2, d_bp.W2t, output_dim, h1);
    transpose(d_cache.X, d_bp.Xt, input_dim, size);
    myGEMM(d_bp.ydiff, d_bp.a1t, d_grad.dW2, &alpha, &beta, output_dim, h1, size);
    naive_sum(d_bp.ydiff, d_grad.db2, output_dim, size);
    // compute partial w.r.t a1
    myGEMM(d_bp.W2t, d_bp.ydiff, d_bp.dz1, &alpha, &beta, h1, size ,output_dim);
    get_dz1(d_bp.dz1, d_cache.a1, h1, size);
    myGEMM(d_bp.dz1, d_bp.Xt, d_grad.dW1, &alpha, &beta, h1, input_dim, size);
    naive_sum(d_bp.dz1, d_grad.db1, h1, size);

    // Add regularization terms to the grads
    matadd(d_grad.dW1, d_params.W1, d_grad.dW1, h1, input_dim, 1.0, reg/(double)num_procs);
    matadd(d_grad.dW2, d_params.W2, d_grad.dW2, output_dim, h1, 1.0, reg/(double)num_procs);
}



void free_all_CUDA(raw_params &d_params, raw_cache &d_cache, raw_grad &d_grad)
{
    cudaFree(d_params.W1);
    cudaFree(d_params.W2);
    cudaFree(d_params.b1);
    cudaFree(d_params.b2);
    cudaFree(d_cache.X);
    cudaFree(d_cache.z1);
    cudaFree(d_cache.a1);
    cudaFree(d_cache.z2);
    cudaFree(d_cache.y);
    cudaFree(d_cache.yhat);
    cudaFree(d_grad.dW1);
    cudaFree(d_grad.dW2);
    cudaFree(d_grad.db1);
    cudaFree(d_grad.db2);
}

