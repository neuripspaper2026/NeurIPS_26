<<<CODE>>>
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

    for (int i = 0; i < rows; i++) {
        iN[i] = i - 1;
        iS[i] = i + 1;
    }
    for (int j = 0; j < cols; j++) {
        jW[j] = j - 1;
        jE[j] = j + 1;
    }
    iN[0] = 0;
    iS[rows - 1] = rows - 1;
    jW[0] = 0;
    jE[cols - 1] = cols - 1;

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
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum,sum2) num_threads(nthreads)
#endif
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

#ifdef _OPENMP
#pragma omp parallel for private(i,j,k,Jc,dN,dS,dW,dE,G2,L,num,den,qsqr) num_threads(nthreads)
#endif
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                k = i * cols + j;
                Jc = J[k];

                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;

                G2 = (dN[k]*dN[k] + dS[k]*dS[k] + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc * Jc);
                L  = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

                num  = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
                den  = 1.0f + (0.25f * L);
                qsqr = num / (den * den);

                den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
                c[k] = 1.0f / (1.0f + den);

                if (c[k] < 0.0f) c[k] = 0.0f;
                else if (c[k] > 1.0f) c[k] = 1.0f;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(i,j,k,cN,cS,cW,cE,D) num_threads(nthreads)
#endif
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                k = i * cols + j;

                cN = c[k];
                cS = c[iS[i] * cols + j];
                cW = c[k];
                cE = c[i * cols + jE[j]];

                D  = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                J[k] = J[k] + 0.25f * lambda * D;
            }
        }
#ifdef ITERATION
    }
#endif

    // === 最终结果输出到文件 ===
    if (output_path == NULL) {
        fprintf(stderr, "No output file provided.\n");
        // 不中断程序，但提示
    } else {
        FILE* fout = fopen(output_path, "w");
        if (!fout) {
            fprintf(stderr, "Failed to open output file: %s\n", output_path);
            // 出错则退出
