

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include <time.h>

// includes, kernels
#include "../EX3_optimized_codes/backprop_cuda_kernel_claude_v1.cu"
#include "backprop.h"

#include "../../common_rodinia/cuda/profile_main.h"

////////////////////////////////////////////////////////////////////////////////

extern "C" void bpnn_layerforward(float *l1, float *l2, float **conn, int n1,
                                  int n2);

extern "C" void bpnn_output_error(float *delta, float *target, float *output,
                                  int nj, float *err);

extern "C" void bpnn_hidden_error(float *delta_h, int nh, float *delta_o,
                                  int no, float **who, float *hidden,
                                  float *err);

extern "C" void bpnn_adjust_weights(float *delta, int ndelta, float *ly,
                                    int nly, float **w, float **oldw);


extern "C" int run(int argc, char **argv);

extern "C" float **alloc_2d_dbl(int m, int n);

extern "C" float squash(float x);

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

// Timing variables
double total_kernel_time = 0.0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    struct timespec main_start, main_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);
    
    run(argc, argv);

    if (getenv("PROFILE")) {
        // warm up
        for (int i = 0; i < 5; i++)
            run(argc, argv);

        checkCudaErrors(cudaProfilerStart());
        nvtxRangePushA("host");

        run(argc, argv);

        nvtxRangePop();
        checkCudaErrors(cudaProfilerStop());
    }

    // Timing output
    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double main_time = (main_end.tv_sec - main_start.tv_sec) + 
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
    
    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }
    
    fprintf(timing_file, "KERNEL_TIME: %.9f\n", total_kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
    
    if (timing_file != stderr)
        fclose(timing_file);

    return EXIT_SUCCESS;
}


extern "C" void bpnn_train_cuda(BPNN *net, float *eo, float *eh) {
    int in, hid, out;
    float out_err, hid_err;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

#ifdef GPU
    int m = 0;
    float *input_hidden_cuda;
    float *input_cuda;
    float *output_hidden_cuda;
    float *partial_sum;
    float *hidden_partial_sum;
    float *hidden_delta_cuda;
    float *input_prev_weights_cuda;
    float sum;
    float *input_weights_one_dim;
    float *input_weights_prev_one_dim;
    num_blocks = in / 16;
    dim3 grid(1, num_blocks);
    dim3 threads(16, 16);

    input_weights_one_dim =
        (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
    input_weights_prev_one_dim =
        (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
    partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));

    // this preprocessing stage is added to correct the bugs of wrong memcopy
    // using two-dimensional net->inputweights
    for (int k = 0; k <= in; k++) {
        for (int j = 0; j <= hid; j++) {
            input_weights_one_dim[m] = net->input_weights[k][j];
            input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
            m++;
        }
    }

    cudaMalloc((void **)&input_cuda, (in + 1) * sizeof(float));
    cudaMalloc((void **)&output_hidden_cuda, (hid + 1) * sizeof(float));
    cudaMalloc((void **)&input_hidden_cuda,
               (in + 1) * (hid + 1) * sizeof(float));
    cudaMalloc((void **)&hidden_partial_sum,
               num_blocks * WIDTH * sizeof(float));


#endif

#ifdef CPU

    printf("Performing CPU computation\n");
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights,
                      in, hid);

#endif

#ifdef GPU

    printf("Performing GPU computation\n");

    // printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);

    cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
               (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

    struct timespec kernel_start_1, kernel_end_1;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start_1);

    PROFILE((
        bpnn_layerforward_CUDA<<<grid, threads>>>(input_cuda, output_hidden_cuda,
                                                  input_hidden_cuda,
                                                  hidden_partial_sum, in, hid)
    ));

    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end_1);
    double kernel_time_1 = (kernel_end_1.tv_sec - kernel_start_1.tv_sec) + 
                           (kernel_end_1.tv_nsec - kernel_start_1.tv_nsec) / 1e9;
    total_kernel_time += kernel_time_1;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(partial_sum, hidden_partial_sum,
               num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    for (int j = 1; j <= hid; j++) {
        sum = 0.0;
        for (int k = 0; k < num_blocks; k++) {
            sum += partial_sum[k * hid + j - 1];
        }
        sum += net->input_weights[0][j];
        net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    }
#endif

    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                      hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                      &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                      net->hidden_weights, net->hidden_units, &hid_err);
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                        net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                        net->input_weights, net->input_prev_weights);

#endif


#ifdef GPU

    cudaMalloc((void **)&hidden_delta_cuda, (hid + 1) * sizeof(float));
    cudaMalloc((void **)&input_prev_weights_cuda,
               (in + 1) * (hid + 1) * sizeof(float));

    cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim,
               (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
               (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

    struct timespec kernel_start_2, kernel_end_2;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start_2);

    PROFILE((
        bpnn_adjust_weights_cuda<<<grid, threads>>>(
            hidden_delta_cuda, hid, input_cuda, in, input_hidden_cuda,
            input_prev_weights_cuda)
    ));
    
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &kernel_end_2);
    double kernel_time_2 = (kernel_end_2.tv_sec - kernel_start_2.tv_sec) + 
                           (kernel_end_2.tv_nsec - kernel_start_2.tv_nsec) / 1e9;
    total_kernel_time += kernel_time_2;

    cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(input_weights_one_dim, input_hidden_cuda,
               (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input_cuda);
    cudaFree(output_hidden_cuda);
    cudaFree(input_hidden_cuda);
    cudaFree(hidden_partial_sum);
    cudaFree(input_prev_weights_cuda);
    cudaFree(hidden_delta_cuda);

    free(partial_sum);
    free(input_weights_one_dim);
    free(input_weights_prev_one_dim);

#endif
}
