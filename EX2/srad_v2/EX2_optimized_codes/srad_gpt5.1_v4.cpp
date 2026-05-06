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

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int kk = 0; kk < size_I; kk++) {
        J[kk] = (float)expf(I[kk]);
    }

    printf("Start the SRAD main loop\n");

#ifdef ITERATION
    for (iter = 0; iter < niter; iter++) {
#endif
        sum = 0.0f;
        sum2 = 0.0f;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum,sum2) schedule(static)
#endif
        for (i = r1; i <= r2; i++) {
            int base = i * cols;
            for (j = c1; j <= c2; j++) {
                tmp = J[base + j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }

        meanROI = sum / (float)size_R;
        varROI = (sum2 / (float)size_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

#ifdef _OPENMP
#pragma omp parallel for private(j,k,Jc,G2,L,num,den,qsqr) schedule(static)
#endif
        for (i = 0; i < rows; i++) {
            int in = iN[i];
            int is = iS[i];
            int row_base = i * cols;
            int in_base = in * cols;
            int is_base = is * cols;
            for (j = 0; j < cols; j++) {
                int jw = jW[j];
                int je = jE[j];
                k = row_base + j;
                Jc = J[k];

                float jn = J[in_base + j];
                float js = J[is_base + j];
                float jwv = J[row_base + jw];
                float jev = J[row_base + je];

                float dn = jn - Jc;
                float ds = js - Jc;
                float dw = jwv - Jc;
                float de = jev - Jc;

                dN[k] = dn;
                dS[k] = ds;
                dW[k] = dw;
                dE[k] = de;

                float jc2 = Jc * Jc;
                G2 = (dn * dn + ds * ds + dw * dw + de * de) / jc2;
                L  = (dn + ds + dw + de) / Jc;

                float L2 = L * L;
                num  = 0.5f * G2 - (1.0f / 16.0f) * L2;
                den  = 1.0f + 0.25f * L;
                float den2 = den * den;
                qsqr = num / den2;

                den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
                float ck = 1.0f / (1.0f + den);

                if (ck < 0.0f) ck = 0.0f;
                else if (ck > 1.0f) ck = 1.0f;
                c[k] = ck;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(j,k,cN,cS,cW,cE,D) schedule(static)
#endif
        for (i = 0; i < rows; i++) {
            int is = iS[i];
            int row_base = i * cols;
            int is_base = is * cols;
            for (j = 0; j < cols; j++) {
                int je = jE[j];
                k = row_base + j;

                cN = c[k];
                cS = c[is_base + j];
                cW = c[k];
                cE = c[row_base + je];

                D  = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                J[k] = J[k] + 0.25f * lambda * D;
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
            for (i = 0; i < rows; i++) {
                int base = i * cols;
                for (j = 0; j < cols; j++) {
                    fprintf(fout, "%.8f", J[base + j]);
                    if (j + 1 < cols) fputc(' ', fout);
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
// printf("%g ", I[i * cols + j]);
#endif
        }
#ifdef OUTPUT
// printf("\n");
#endif
    }
}
