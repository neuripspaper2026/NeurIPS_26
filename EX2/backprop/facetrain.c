#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include <string.h>
#include <unistd.h>

extern void bpnn_train_kernel(BPNN *net, float *eo, float *eh);
extern int load(BPNN *net);

int layer_size = 0;
static const char *g_output_file = NULL;

void backprop_face() {
    BPNN *net;
    int i;
    float out_err, hid_err;
    net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
    printf("Input layer size : %d\n", layer_size);
    load(net);
    // entering the training kernel, only one iteration
    printf("Starting training kernel\n");
    bpnn_train_kernel(net, &out_err, &hid_err);
    if (g_output_file != NULL) {
        bpnn_save(net, g_output_file);
    } else if (getenv("OUTPUT")) {
        bpnn_save(net, "output.dat");
    }
    bpnn_free(net);
    printf("Training done\n");
}

int setup(int argc, char **argv) {
    if (argc != 2 && !(argc == 4 && strcmp(argv[2], "-o") == 0)) {
        fprintf(stderr, "usage: backprop <num of input elements> [-o <output_file>]\n");
        exit(1);
    }

    layer_size = atoi(argv[1]);
    if (argc == 4) {
        g_output_file = argv[3];
    }
    if (layer_size % 16 != 0) {
        fprintf(stderr, "The number of input points must be divided by 16\n");
        exit(1);
    }

    int seed = 7;
    bpnn_initialize(seed);
    backprop_face();

    exit(0);
}