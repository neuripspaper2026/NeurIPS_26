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

    int rows, cols, size_I, size_R, niter = 10;
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

    {
        const int total = size_I;
        for (int k = 0; k < total; ++k) {
            J[k] = (float)expf(I[k]);
        }
    }

    printf("Start the SRAD main loop\n");

#ifdef ITERATION
    for (int iter = 0; iter < niter; iter++) {
#endif
        sum = 0.0f;
        sum2 = 0.0f;
        {
            const int col_stride = cols;
            for (i = r1; i <= r2; i++) {
                int idx = i * col_stride + c1;
                for (j = c1; j <= c2; j++, idx++) {
                    tmp = J[idx];
                    sum += tmp;
                    sum2 += tmp * tmp;
                }
            }
        }
        meanROI = sum / (float)size_R;
        varROI = (sum2 / (float)size_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        {
            const int col_stride = cols;
            for (i = 0; i < rows; i++) {
                const int i_index = i * col_stride;
                const int in_index = iN[i] * col_stride;
                const int is_index = iS[i] * col_stride;
                for (j = 0; j < cols; j++) {
                    const int k = i_index + j;
                    Jc = J[k];

                    const float jn = J[in_index + j];
                    const float js = J[is_index + j];
                    const float jw = J[i_index + jW[j]];
                    const float je = J[i_index + jE[j]];

                    const float dNk = jn - Jc;
                    const float dSk = js - Jc;
                    const float dWk = jw - Jc;
                    const float dEk = je - Jc;

                    dN[k] = dNk;
                    dS[k] = dSk;
                    dW[k] = dWk;
                    dE[k] = dEk;

                    G2 = (dNk * dNk + dSk * dSk + dWk * dWk + dEk * dEk) / (Jc * Jc);
                    L  = (dNk + dSk + dWk + dEk) / Jc;

                    num  = 0.5f * G2 - (0.0625f * L * L);
                    den  = 1.0f + 0.25f * L;
                    qsqr = num / (den * den);

                    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
                    {
                        float ck = 1.0f / (1.0f + den);
                        if (ck < 0.0f) ck = 0.0f;
                        else if (ck > 1.0f) ck = 1.0f;
                        c[k] = ck;
                    }
                }
            }
        }

        {
            const int col_stride = cols;
            const float lambda_quarter = 0.25f * lambda;
            for (i = 0; i < rows; i++) {
                const int i_index = i * col_stride;
                const int is_index = iS[i] * col_stride;
                for (j = 0; j < cols; j++) {
                    const int k = i_index + j;

                    cN = c[k];
                    cS = c[is_index + j];
                    cW = c[k];
                    cE = c[i_index + jE[j]];

                    D  = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                    J[k] = J[k] + lambda_quarter * D;
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
        if (!fout) {
            fprintf(stderr, "Failed to open output file: %s\n", output_path);
            free(I); free(J); free(iN); free(iS); free(jW); free(jE);
            free(dN); free(dS); free(dW); free(dE); free(c);
            return 2;
        }

        for (i = 0; i < rows; i++) {
            int idx = i * cols;
            for (j = 0; j < cols; j++, idx++) {
                fprintf(fout, "%.8f", J[idx]);
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
        int idx = i * cols;
        for (int j = 0; j < cols; j++, idx++) {
            I[idx] = rand() / (float)RAND_MAX;
#ifdef OUTPUT
#endif
        }
#ifdef OUTPUT
#endif
    }
}
