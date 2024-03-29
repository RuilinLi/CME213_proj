#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {
    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    // My code
    // Allocate host memory on CPU
    struct raw_grad h_grad; // host grad
    int max_num_img_per_node = (batch_size + num_procs-1)/num_procs;
    // double *X_local = (double*)malloc(sizeof(double)*max_num_img_per_node*nn.H[0]);
    // double *y_local = (double*)malloc(sizeof(double)*max_num_img_per_node*nn.H[2]);
    // for(int layer = 0; layer < nn.H.size()-1; layer++){
    //     h_grad.dW.emplace_back(nn.H[layer+1], nn.H[layer]);
    //     h_grad.db.emplace_back(nn.H[layer+1]);
    // }
    double *X_local;
    double *y_local;
    cudaHostAlloc((void**)&X_local, sizeof(double)*max_num_img_per_node*nn.H[0],cudaHostAllocPortable);
    cudaHostAlloc((void**)&y_local, sizeof(double)*max_num_img_per_node*nn.H[2],cudaHostAllocPortable);

    cudaHostAlloc((void**)&h_grad.dW1, sizeof(double)*nn.H[0]*nn.H[1],cudaHostAllocPortable);
    cudaHostAlloc((void**)&h_grad.dW2, sizeof(double)*nn.H[1]*nn.H[2],cudaHostAllocPortable);
    cudaHostAlloc((void**)&h_grad.db1, sizeof(double)*nn.H[1],cudaHostAllocPortable);
    cudaHostAlloc((void**)&h_grad.db2, sizeof(double)*nn.H[2],cudaHostAllocPortable);

    raw_params d_params;
    raw_cache d_cache;
    raw_grad d_grad;
    raw_bp d_bp;
    allocate_device_memory(d_params, d_cache, d_grad, d_bp, nn.H, max_num_img_per_node);
    // MPI paramters setup
    int *batch_size_by_rank = (int*)malloc(num_procs*sizeof(int));
    int *sendcounts_X = (int*)malloc(num_procs*sizeof(int));
    int *sendcounts_y = (int*)malloc(num_procs*sizeof(int));
    int *displs_X = (int*)malloc(num_procs*sizeof(int));
    int *displs_y = (int*)malloc(num_procs*sizeof(int));
    int size_each_node;
    int starting_location;
    int current_batch_size;
    int batch_size_node;

    // Sync the weights, seems that  the nns have have the same initialization, so this is
    // not needed
    // MPI_Bcast(nn.W[0].memptr(), nn.H[0]*nn.H[1], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Bcast(nn.W[1].memptr(), nn.H[1]*nn.H[2], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Bcast(nn.b[0].memptr(), nn.H[1], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Bcast(nn.b[1].memptr(), nn.H[2], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Copy initial parameters to device
    cudaMemcpy(d_params.W1, nn.W[0].memptr(), sizeof(double) * nn.H[0] * nn.H[1], cudaMemcpyHostToDevice);
    cudaMemcpy(d_params.W2, nn.W[1].memptr(), sizeof(double) * nn.H[1] * nn.H[2], cudaMemcpyHostToDevice);
    cudaMemcpy(d_params.b1, nn.b[0].memptr(), sizeof(double) * nn.H[1], cudaMemcpyHostToDevice);
    cudaMemcpy(d_params.b2, nn.b[1].memptr(), sizeof(double) * nn.H[2], cudaMemcpyHostToDevice);


    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;
    cudaStream_t mystream[4];
    for(int si =0; si<4; si++){
        cudaStreamCreate(&mystream[si]);
    }

    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;
        for(int batch = 0; batch < num_batches; ++batch) {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */
             // First scatter the data
            starting_location = batch*batch_size;
            current_batch_size = std::min(batch_size, N-starting_location);
            size_each_node = (current_batch_size + num_procs - 1)/num_procs;
            for (int i = 0; i<num_procs; ++i){
                batch_size_by_rank[i] = std::min(size_each_node, std::max(0, current_batch_size -i*size_each_node));
            }
            batch_size_node = batch_size_by_rank[rank];
            for (int i = 0; i < num_procs; ++i){
                sendcounts_X[i] = batch_size_by_rank[i] * nn.H[0];
                sendcounts_y[i] = batch_size_by_rank[i] * nn.H[2];
                displs_X[i] = i*size_each_node*nn.H[0];
                displs_y[i] = i*size_each_node*nn.H[2];
            }

            MPI_Scatterv(X.colptr(starting_location), sendcounts_X, displs_X, 
                          MPI_DOUBLE, X_local, batch_size_node*nn.H[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

            send_data_to_device(X_local, d_cache.X, batch_size_node, nn.H[0], mystream[0]);

            forward_pass(d_params, d_cache, nn.H[0], nn.H[1], nn.H[2], batch_size_node, mystream);

            MPI_Scatterv(y.colptr(starting_location), sendcounts_y, displs_y, 
                          MPI_DOUBLE, y_local, batch_size_node*nn.H[2], MPI_DOUBLE, 0, MPI_COMM_WORLD);
            send_data_to_device(y_local, d_cache.y, batch_size_node, nn.H[2], mystream[1]);
            cudaDeviceSynchronize();
            backward_pass(d_params,d_cache,d_grad, d_bp,nn.H[0], nn.H[1],nn.H[2], reg, batch_size_node, current_batch_size, num_procs, mystream);

            
            // Send the gradient from GPU to host
            cudaMemcpyAsync(h_grad.dW1, d_grad.dW1, sizeof(double) * nn.H[0] * nn.H[1], cudaMemcpyDeviceToHost, mystream[0]);
            cudaMemcpyAsync(h_grad.dW2, d_grad.dW2, sizeof(double) * nn.H[2] * nn.H[1], cudaMemcpyDeviceToHost, mystream[1]);
            cudaMemcpyAsync(h_grad.db1, d_grad.db1, sizeof(double) * nn.H[1], cudaMemcpyDeviceToHost,  mystream[2]);
            cudaMemcpyAsync(h_grad.db2, d_grad.db2, sizeof(double) * nn.H[2], cudaMemcpyDeviceToHost, mystream[3]);

            cudaDeviceSynchronize();
            MPI_Allreduce(MPI_IN_PLACE, h_grad.dW1, nn.H[0]*nn.H[1], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
            cudaMemcpyAsync(d_grad.dW1, h_grad.dW1, sizeof(double) * nn.H[0] * nn.H[1], cudaMemcpyHostToDevice, mystream[0]);

            MPI_Allreduce(MPI_IN_PLACE, h_grad.dW2, nn.H[1]*nn.H[2], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cudaMemcpyAsync(d_grad.dW2, h_grad.dW2, sizeof(double) * nn.H[2] * nn.H[1], cudaMemcpyHostToDevice, mystream[1]);

            MPI_Allreduce(MPI_IN_PLACE, h_grad.db1, nn.H[1], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cudaMemcpyAsync(d_grad.db1, h_grad.db1, sizeof(double) * nn.H[1], cudaMemcpyHostToDevice,mystream[2]);

            MPI_Allreduce(MPI_IN_PLACE, h_grad.db2, nn.H[2], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cudaMemcpyAsync(d_grad.db2, h_grad.db2, sizeof(double) * nn.H[2], cudaMemcpyHostToDevice, mystream[3]);

            // Seems like gradient descent should be done on the GPUs, so we do this
            // Send the gradient from host to GPU

            gradient_descent(d_grad, d_params, learning_rate, nn.H[0], nn.H[1], nn.H[2], mystream);

            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }
            iter++;
        }
    }
    // Copy the result back to host 
    cudaMemcpy(nn.W[0].memptr(), d_params.W1, sizeof(double) * nn.H[0] * nn.H[1], cudaMemcpyDeviceToHost);
    cudaMemcpy(nn.W[1].memptr(), d_params.W2, sizeof(double) * nn.H[1] * nn.H[2], cudaMemcpyDeviceToHost);
    cudaMemcpy(nn.b[0].memptr(), d_params.b1, sizeof(double) * nn.H[1], cudaMemcpyDeviceToHost);
    cudaMemcpy(nn.b[1].memptr(), d_params.b2, sizeof(double) * nn.H[2], cudaMemcpyDeviceToHost);

    cudaFreeHost(X_local);
    cudaFreeHost(y_local);
    cudaFreeHost(h_grad.dW1);
    cudaFreeHost(h_grad.dW2);
    cudaFreeHost(h_grad.db1);
    cudaFreeHost(h_grad.db2);
    for (int si = 0; si< 4;++si){
        cudaStreamDestroy(mystream[si]);
    }
    free(sendcounts_X);
    free(sendcounts_y);
    free(displs_X);
    free(displs_y);
    free_all_CUDA(d_params, d_cache, d_grad);

    error_file.close();
}
