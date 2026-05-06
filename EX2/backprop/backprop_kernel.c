#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1,
                              int n2);

extern void bpnn_output_error(float *delta, float *target, float *output,
                              int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
                              float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
                                float **w, float **oldw);


extern int setup(int argc, char **argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    struct timespec main_start, main_end;
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    setup(argc, argv);
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);

    clock_gettime(CLOCK_MONOTONIC, &main_end);

    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);
}


void bpnn_train_kernel(BPNN *net, float *eo, float *eh) {
    int in, hid, out;
    float out_err, hid_err;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    printf("Performing CPU computation\n");
    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights,
                      in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                      hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                      &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                      net->hidden_weights, net->hidden_units, &hid_err);
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                        net->hidden_weights, net->hidden_prev_weights);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                        net->input_weights, net->input_prev_weights);
}
