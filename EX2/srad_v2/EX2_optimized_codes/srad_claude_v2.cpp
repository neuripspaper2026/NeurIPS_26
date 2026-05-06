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

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        iN[i] = i - 1;
        iS[i] = i + 1;
    }
    #pragma omp parallel for
    for (int j = 0; j < cols; j++) {
        jW[j] = j - 1;
        jE[j] = j + 1;
    }
    iN[0] = 0;
    iS[rows - 1] = rows - 1;
    jW[0] = 0;
    jE[cols - 1] = cols - 1;

    printf("Initializing the input matrix (deterministic)\n");
    random_matrix(I, rows, cols);

    #pragma omp parallel for
    for (k = 0; k < size_I; k++) {
        J[k] = expf(I[k]);
    }

    printf("Start the SRAD main loop\n");

    const float lambda_quarter = 0.25f * lambda;

#ifdef ITERATION
    for (int iter = 0; iter < niter; iter++) {
#endif
        sum = 0.0f;
        sum2 = 0.0f;
        #pragma omp parallel for reduction(+:sum,sum2) collapse(2)
        for (i = r1; i <= r2; i++) {
            for (j = c1; j <= c2; j++) {
                tmp = J[i * cols + j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }
        meanROI = sum / size_R;
        varROI = (sum2 / size_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        #pragma omp parallel for private(i,j,k,Jc,G2,L,num,den,qsqr)
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                k = i * cols + j;
                Jc = J[k];

                float dN_val = J[iN[i] * cols + j] - Jc;
                float dS_val = J[iS[i] * cols + j] - Jc;
                float dW_val = J[i * cols + jW[j]] - Jc;
                float dE_val = J[i * cols + jE[j]] - Jc;

                dN[k] = dN_val;
                dS[k] = dS_val;
                dW[k] = dW_val;
                dE[k] = dE_val;

                G2 = (dN_val*dN_val + dS_val*dS_val + dW_val*dW_val + dE_val*dE_val) / (Jc * Jc);
                L  = (dN_val + dS_val + dW_val + dE_val) / Jc;

                num  = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
                den  = 1.0f + (0.25f * L);
                qsqr = num / (den * den);

                den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
                float c_val = 1.0f / (1.0f + den);

                if (c_val < 0.0f) c_val = 0.0f;
                else if (c_val > 1.0f) c_val = 1.0f;

                c[k] = c_val;
            }
        }

        #pragma omp parallel for private(i,j,k,cN,cS,cW,cE,D)
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                k = i * cols + j;

                cN = c[k];
                cS = c[iS[i] * cols + j];
                cW = c[k];
                cE = c[i * cols + jE[j]];

                D  = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                J[k] = J[k] + lambda_quarter * D;
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
            free(dN); free(dS); free(dW); free(dE); free(c);
            return 2;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(fout, "%.8f", J[i * cols + j]);
                if (j + 1 < cols) fputc(' ', fout);
            }
            fputc('\n', fout);
        }
        fclose(fout);
        printf("Result written to: %s\n", output_path);
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
        for (int j = 0; j < cols; j++) {
            I[i * cols + j] = rand() / (float)RAND_MAX;
#ifdef OUTPUT
#endif
        }
#ifdef OUTPUT
#endif
    }
}
