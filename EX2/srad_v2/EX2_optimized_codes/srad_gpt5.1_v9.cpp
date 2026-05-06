#define ITERATION
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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
        (void)nthreads;
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

    for (int ii = 0; ii < rows; ii++) {
        iN[ii] = ii - 1;
        iS[ii] = ii + 1;
    }
    for (int jj = 0; jj < cols; jj++) {
        jW[jj] = jj - 1;
        jE[jj] = jj + 1;
    }
    iN[0] = 0;
    iS[rows - 1] = rows - 1;
    jW[0] = 0;
    jE[cols - 1] = cols - 1;

    printf("Initializing the input matrix (deterministic)\n");
    random_matrix(I, rows, cols);

    for (k = 0; k < size_I; k++) {
        J[k] = (float)expf(I[k]);
    }

    printf("Start the SRAD main loop\n");

#ifdef ITERATION
    for (iter = 0; iter < niter; iter++) {
#endif
        sum = 0.0f;
        sum2 = 0.0f;
        {
            const int c1_loc = c1;
            const int c2_loc = c2;
            const int r1_loc = r1;
            const int r2_loc = r2;
            const int cols_loc = cols;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum,sum2) schedule(static)
#endif
            for (i = r1_loc; i <= r2_loc; i++) {
                int idx = i * cols_loc + c1_loc;
                for (j = c1_loc; j <= c2_loc; j++, idx++) {
                    tmp = J[idx];
                    sum += tmp;
                    sum2 += tmp * tmp;
                }
            }
        }
        meanROI = sum / size_R;
        varROI = (sum2 / size_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        {
            const int rows_loc = rows;
            const int cols_loc = cols;
            const float q0sqr_loc = q0sqr;
            const int * __restrict iN_loc = iN;
            const int * __restrict iS_loc = iS;
            const int * __restrict jW_loc = jW;
            const int * __restrict jE_loc = jE;
            float * __restrict J_loc = J;
            float * __restrict dN_loc = dN;
            float * __restrict dS_loc = dS;
            float * __restrict dW_loc = dW;
            float * __restrict dE_loc = dE;
            float * __restrict c_loc  = c;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(j,k,Jc,G2,L,num,den,qsqr)
#endif
            for (i = 0; i < rows_loc; i++) {
                int in = iN_loc[i] * cols_loc;
                int is = iS_loc[i] * cols_loc;
                int row = i * cols_loc;
                for (j = 0; j < cols_loc; j++) {
                    k = row + j;
                    Jc = J_loc[k];

                    float jn = J_loc[in + j];
                    float js = J_loc[is + j];
                    float jw = J_loc[row + jW_loc[j]];
                    float je = J_loc[row + jE_loc[j]];

                    float dNk = jn - Jc;
                    float dSk = js - Jc;
                    float dWk = jw - Jc;
                    float dEk = je - Jc;

                    dN_loc[k] = dNk;
                    dS_loc[k] = dSk;
                    dW_loc[k] = dWk;
                    dE_loc[k] = dEk;

                    float Jc2 = Jc * Jc;
                    G2 = (dNk * dNk + dSk * dSk + dWk * dWk + dEk * dEk) / Jc2;
                    L  = (dNk + dSk + dWk + dEk) / Jc;

                    num  = 0.5f * G2 - 0.0625f * (L * L);
                    den  = 1.0f + 0.25f * L;
                    qsqr = num / (den * den);

                    den = (qsqr - q0sqr_loc) / (q0sqr_loc * (1.0f + q0sqr_loc));
                    float ck = 1.0f / (1.0f + den);

                    if (ck < 0.0f) ck = 0.0f;
                    else if (ck > 1.0f) ck = 1.0f;

                    c_loc[k] = ck;
                }
            }
        }

        {
            const int rows_loc = rows;
            const int cols_loc = cols;
            const int * __restrict iS_loc = iS;
            const int * __restrict jE_loc = jE;
            float * __restrict J_loc = J;
            float * __restrict dN_loc = dN;
            float * __restrict dS_loc = dS;
            float * __restrict dW_loc = dW;
            float * __restrict dE_loc = dE;
            float * __restrict c_loc  = c;
            const float lambda_loc = lambda;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(j,k,cN,cS,cW,cE,D)
#endif
            for (i = 0; i < rows_loc; i++) {
                int is = iS_loc[i] * cols_loc;
                int row = i * cols_loc;
                for (j = 0; j < cols_loc; j++) {
                    k = row + j;

                    cN = c_loc[k];
                    cS = c_loc[is + j];
                    cW = c_loc[k];
                    cE = c_loc[row + jE_loc[j]];

                    D  = cN * dN_loc[k] + cS * dS_loc[k] + cW * dW_loc[k] + cE * dE_loc[k];
                    J_loc[k] = J_loc[k] + 0.25f * lambda_loc * D;
                }
            }
        }
#ifdef ITERATION
    }
#endif

    if (output_path == NULL) {
        fprintf(stderr, "No output file provided.\n");
    } else {
        FILE* fout = fopen(output_path, "w");
        int file_ok = 1;
        if (!fout) {
            fprintf(stderr, "Failed to open output file: %s\n", output_path);
            file_ok = 0;
        }
        if (file_ok) {
            for (int ii = 0; ii < rows; ii++) {
                int base = ii * cols;
                for (int jj = 0; jj < cols; jj++) {
                    fprintf(fout, "%.8f", J[base + jj]);
                    if (jj + 1 < cols) fputc(' ', fout);
                }
                fputc('\n', fout);
            }
            fclose(fout);
            printf("Result written to: %s\n", output_path);
        }
    }

    printf("Computation Done\n");

    free(I);
    free(J);
    free(iN);
    free(iS);
    free(jW);
    free(jE);
    free(dN);
    free(dS);
    free(dW);
    free(dE);
    free(c);

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);

    return 0;
}

void random_matrix(float *I, int rows, int cols) {

    srand(7);

    for (int i = 0; i < rows; i++) {
        int base = i * cols;
        for (int j = 0; j < cols; j++) {
            I[base + j] = rand() / (float)RAND_MAX;
#ifdef OUTPUT
#endif
        }
#ifdef OUTPUT
#endif
    }
}
