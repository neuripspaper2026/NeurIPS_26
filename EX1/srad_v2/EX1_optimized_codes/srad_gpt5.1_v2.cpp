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

    int rows, cols, size_I, size_R, niter = 10, k;
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

    I = (float *)malloc((size_t)size_I * sizeof(float));
    J = (float *)malloc((size_t)size_I * sizeof(float));
    c = (float *)malloc((size_t)size_I * sizeof(float));

    iN = (int *)malloc((size_t)rows * sizeof(int));
    iS = (int *)malloc((size_t)rows * sizeof(int));
    jW = (int *)malloc((size_t)cols * sizeof(int));
    jE = (int *)malloc((size_t)cols * sizeof(int));

    dN = (float *)malloc((size_t)size_I * sizeof(float));
    dS = (float *)malloc((size_t)size_I * sizeof(float));
    dW = (float *)malloc((size_t)size_I * sizeof(float));
    dE = (float *)malloc((size_t)size_I * sizeof(float));

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
        J[k] = (float)expf(I[k]);
    }

    printf("Start the SRAD main loop\n");

#ifdef ITERATION
    for (int iter = 0; iter < niter; iter++) {
#endif
        sum = 0.0f;
        sum2 = 0.0f;

        for (i = r1; i <= r2; i++) {
            const int row_offset = i * cols;
            for (j = c1; j <= c2; j++) {
                tmp = J[row_offset + j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }

        meanROI = sum / (float)size_R;
        varROI = (sum2 / (float)size_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        for (i = 0; i < rows; i++) {
            const int i_cols = i * cols;
            const int iN_i = iN[i];
            const int iS_i = iS[i];
            const int iN_cols = iN_i * cols;
            const int iS_cols = iS_i * cols;
            for (j = 0; j < cols; j++) {
                const int idx = i_cols + j;
                Jc = J[idx];

                const int jW_j = jW[j];
                const int jE_j = jE[j];

                const int idxN = iN_cols + j;
                const int idxS = iS_cols + j;
                const int idxW = i_cols + jW_j;
                const int idxE = i_cols + jE_j;

                const float dn = J[idxN] - Jc;
                const float ds = J[idxS] - Jc;
                const float dw = J[idxW] - Jc;
                const float de = J[idxE] - Jc;

                dN[idx] = dn;
                dS[idx] = ds;
                dW[idx] = dw;
                dE[idx] = de;

                G2 = (dn*dn + ds*ds + dw*dw + de*de) / (Jc * Jc);
                L  = (dn + ds + dw + de) / Jc;

                num  = (0.5f * G2) - (0.0625f * (L * L));
                den  = 1.0f + (0.25f * L);
                qsqr = num / (den * den);

                den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
                float ck = 1.0f / (1.0f + den);

                if (ck < 0.0f) ck = 0.0f;
                else if (ck > 1.0f) ck = 1.0f;
                c[idx] = ck;
            }
        }

        for (i = 0; i < rows; i++) {
            const int i_cols = i * cols;
            const int iS_i = iS[i];
            const int iS_cols = iS_i * cols;
            for (j = 0; j < cols; j++) {
                const int idx = i_cols + j;
                const int jE_j = jE[j];
                const int idxS = iS_cols + j;
                const int idxE = i_cols + jE_j;

                cN = c[idx];
                cS = c[idxS];
                cW = c[idx];
                cE = c[idxE];

                D = cN * dN[idx] + cS * dS[idx] + cW * dW[idx] + cE * dE[idx];
                J[idx] = J[idx] + 0.25f * lambda * D;
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
            return 2;
        }

        for (i = 0; i < rows; i++) {
            const int i_cols = i * cols;
            for (j = 0; j < cols; j++) {
                fprintf(fout, "%.8f", J[i_cols + j]);
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
    double kernel_time = (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
                         (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (double)(main_end.tv_sec - main_start.tv_sec) +
                       (double)(main_end.tv_nsec - main_start.tv_nsec) / 1e9;

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
        const int row_offset = i * cols;
        for (int j = 0; j < cols; j++) {
            I[row_offset + j] = rand() / (float)RAND_MAX;
#ifdef OUTPUT
#endif
        }
#ifdef OUTPUT
#endif
    }
}
