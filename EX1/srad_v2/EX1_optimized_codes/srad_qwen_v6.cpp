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

    // 现在需要 10 个参数 + 程序名 => argc == 11
    if (argc == 11) {
        rows = atoi(argv[1]); // number of rows in the domain
        cols = atoi(argv[2]); // number of cols in the domain
        if ((rows % 16 != 0) || (cols % 16 != 0)) {
            fprintf(stderr, "rows and cols must be multiples of 16\n");
            exit(1);
        }
        r1 = atoi(argv[3]); // y1 position of the speckle
        r2 = atoi(argv[4]); // y2 position of the speckle
        c1 = atoi(argv[5]); // x1 position of the speckle
        c2 = atoi(argv[6]); // x2 position of the speckle
        nthreads = atoi(argv[7]); // number of threads
        lambda = atof(argv[8]); // Lambda value
        niter = atoi(argv[9]); // number of iterations
        output_path = argv[10]; // 输出文件路径
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

    // Precompute boundary conditions
    for (int i = 0; i < rows; i++) {
        iN[i] = (i == 0) ? 0 : i - 1;
        iS[i] = (i == rows - 1) ? rows - 1 : i + 1;
    }
    for (int j = 0; j < cols; j++) {
        jW[j] = (j == 0) ? 0 : j - 1;
        jE[j] = (j == cols - 1) ? cols - 1 : j + 1;
    }

    printf("Initializing the input matrix (deterministic)\n");
    random_matrix(I, rows, cols); // 已经是确定性填充

    for (k = 0; k < size_I; k++) {
        J[k] = (float)exp(I[k]);
    }

    printf("Start the SRAD main loop\n");

#ifdef ITERATION
    for (int iter = 0; iter < niter; iter++) {
#endif
        sum = 0.0f;
        sum2 = 0.0f;
        for (i = r1; i <= r2; i++) {
            float* row = &J[i * cols];
            for (j = c1; j <= c2; j++) {
                tmp = row[j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }
        meanROI = sum / size_R;
        varROI = (sum2 / size_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        // Loop unrolling for computing dN, dS, dW, dE and c
        for (i = 0; i < rows; i++) {
            const int base_idx = i * cols;
            const float* const J_row     = &J[base_idx];
            const float* const J_row_north = &J[iN[i] * cols];
            const float* const J_row_south = &J[iS[i] * cols];
            float* const c_row = &c[base_idx];
            float* const dN_row = &dN[base_idx];
            float* const dS_row = &dS[base_idx];
            float* const dW_row = &dW[base_idx];
            float* const dE_row = &dE[base_idx];
            const int jW_0 = jW[0];
            const int jE_last = jE[cols - 1];

            for (j = 0; j < cols; j++) {
                const int idx = base_idx + j;
                Jc = J_row[j];

                dN_row[j] = J_row_north[j] - Jc;
                dS_row[j] = J_row_south[j] - Jc;
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

        // Loop unrolling for updating J
        for (i = 0; i < rows; i++) {
            const int base_idx = i * cols;
            float* const J_row = &J[base_idx];
            const float* const c_row = &c[base_idx];
            const float* const c_row_south = &c[iS[i] * cols];
            const float* const dN_row = &dN[base_idx];
            const float* const dS_row = &dS[base_idx];
            const float* const dW_row = &dW[base_idx];
            const float* const dE_row = &
