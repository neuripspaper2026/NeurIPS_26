<<<CODE>>>
#define ITERATION
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

void random_matrix(float *I, int rows, int cols);

void usage(int argc, char **argv) {
    fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <no. of threads> <lamda> <no. of iter> <output_file>\n",
            argv[0]);
    fprintf(stderr, "\t<rows>           - number of rows (multiple of 16)\n");
    fprintf(stderr, "\t<cols>           - number of cols (multiple of 16)\n");
    fprintf(stderr, "\t<y1>             - y1 value of the speckle\n");
    fprintf(stderr, "\t<y2>             - y2 value of the speckle\n");
    fprintf(stderr, "\t<x1>             - x1 value of the speckle\n");
    fprintf(stderr, "\t<x2>             - x2 value of the speckle\n");
    fprintf(stderr, "\t<no. of threads> - no. of threads\n");
    fprintf(stderr, "\t<lamda>          - lambda (0,1)\n");
    fprintf(stderr, "\t<no. of iter>    - number of iterations\n");
    fprintf(stderr, "\t<output_file>    - path to write the final result matrix J\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    struct timespec main_start, main_end;
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    int rows, cols, size_I, size_R, niter = 10, iter, k;
    float *I, *J, q0sqr, sum, sum2, tmp, meanROI, varROI;
    float Jc, G2, L, num, den, qsqr;
    int *iN, *iS, *jE, *jW;
    float *dN, *dS, *dW, *dE;
    int r1, r2, c1, c2;
    float cN, cS, cW, cE;
    float *c, D;
    float lambda;
    int i, j;
    int nthreads;

    const char* output_path = NULL;

    if (argc == 11) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        if ((rows % 16 != 0) || (cols % 16 != 0)) {
            fprintf(stderr, "rows and cols must be multiples of 16\n");
            exit(1);
        }
        r1 = atoi(argv[3]);
        r2 = atoi(argv[4]);
        c1 = atoi(argv[5]);
        c2 = atoi(argv[6]);
        nthreads = atoi(argv[7]);
        lambda = atof(argv[8]);
        niter = atoi(argv[9]);
        output_path = argv[10];
    } else {
        usage(argc, argv);
    }

    size_I = cols * rows;
    size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

    I = (float *)malloc(size_I * sizeof(float));
    J = (float *)malloc(size_I * sizeof(float));
    c = (float *)malloc(sizeof(float) * size_I);

    iN = (int *)malloc(sizeof(int) * rows);
    iS = (int *)malloc(sizeof(int) * rows);
    jW = (int *)malloc(sizeof(int) * cols);
    jE = (int *)malloc(sizeof(int) * cols);

    dN = (float *)malloc(sizeof(float) * size_I);
    dS = (float *)malloc(sizeof(float) * size_I);
    dW = (float *)malloc(sizeof(float) * size_I);
    dE = (float *)malloc(sizeof(float) * size_I);

    for (i = 0; i < rows; i++) {
        iN[i] = i - 1;
        iS[i] = i + 1;
    }
    for (j = 0; j < cols; j++) {
        jW[j] = j - 1;
        jE[j] = j + 1;
    }
    iN[0] = 0;
    iS[rows - 1] = rows - 1;
    jW[0] = 0;
    jE[cols - 1] = cols - 1;

    printf("Initializing the input matrix (deterministic)\n");
    random_matrix(I, rows, cols);

    for (k = 0; k < size_I; k++) {
        J[k] = (float)exp(I[k]);
    }

    printf("Start the SRAD main loop\n");

#ifdef ITERATION
    for (iter = 0; iter < niter; iter++) {
#endif
        sum = 0.0f;
        sum2 = 0.0f;
        for (i = r1; i <= r2; i++) {
            float *J_row = &J[i * cols];
            for (j = c1; j <= c2; j++) {
                tmp = J_row[j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }
        meanROI = sum / size_R;
        varROI = (sum2 / size_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        for (i = 0; i < rows; i++) {
            int i_cols = i * cols;
            float *J_row = &J[i_cols];
            float *J_iN_row = &J[iN[i] * cols];
            float *J_iS_row = &J[iS[i] * cols];
            float *c_row = &c[i_cols];
            float *dN_row = &dN[i_cols];
            float *dS_row = &dS[i_cols];
            float *dW_row = &dW[i_cols];
            float *dE_row = &dE[i_cols];

            for (j = 0; j < cols; j++) {
                k = i_cols + j;
                Jc = J_row[j];

                dN_row[j] = J_iN_row[j] - Jc;
                dS_row[j] = J_iS_row[j] - Jc;
                dW_row[j] = J_row[jW[j]] - Jc;
                dE_row[j] = J_row[jE[j]] - Jc;

                G2 = (dN_row[j]*dN_row[j] + dS_row[j]*dS_row[j] + dW_row[j]*dW_row[j] + dE_row[j]*dE_row[j]) / (Jc * Jc);
                L  = (dN_row[j] + dS_row[j] + dW_row[j] + dE_row[j]) / Jc;

                num  = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
                den  = 1.0f + (0.25f * L);
                qsqr = num / (den * den);

                den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
                c_row[j] = 1.0f / (1.0f + den);

                if (c_row[j] < 0.0f) c_row[j] = 0.0f;
                else if (c_row[j] > 1.0f) c_row[j] = 1.0f;
            }
        }

        for (i = 0; i < rows; i++) {
            int i_cols = i * cols;
            float *J_row = &J[i_cols];
            float *c_row = &c[i_cols];
            float *dN_row = &dN[i_cols];
            float *dS_row = &dS[i_cols];
            float *dW_row = &dW[i_cols];
            float *dE_row = &dE[i_cols];

            for (j = 0; j < cols; j++) {
                k = i_cols + j;

                cN = c_row[j];
                cS = c[iS[i] * cols + j];
                cW = c_row[j];
                cE = c_row[jE[j]];

                D  = cN * dN_row[j] + cS * dS_row[j] + cW * dW_row[j] + cE * dE_row[j];
                J_row[j] = J_row[j] + 0.25f * lambda * D;
            }
        }
#ifdef ITERATION
    }
#endif

    if (output_path == NULL) {
        fprintf(stderr, "No output file provided.\n");
    } else {
        FILE* fout = fopen(output_path, "w");
        if (!fout) {
            fprintf(stderr, "Failed to open output file: %s\n", output_path);
            free(I); free(J); free(iN); free(iS); free(jW); free(jE);
            free(dN); free
