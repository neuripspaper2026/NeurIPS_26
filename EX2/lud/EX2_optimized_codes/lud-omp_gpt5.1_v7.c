#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;

    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        const int row_i = i * size;

        /* Upper triangular update: A[i][j] */
        for (j = i; j < size; j++) {
            float local_sum = a[row_i + j];

            /* This inner loop is typically short; keep it serial for locality */
            for (k = 0; k < i; k++) {
                const int row_k = k * size;
                local_sum -= a[row_i + k] * a[row_k + j];
            }
            a[row_i + j] = local_sum;
        }

        /* Factor pivot element once to avoid repeated loads */
        const float pivot = a[row_i + i];

        /* Lower triangular update: A[j][i] (parallelizable across rows) */
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(a, size, i, pivot) private(j, k, sum)
#endif
        for (j = i + 1; j < size; j++) {
            const int row_j = j * size;
            float local_sum = a[row_j + i];

            for (k = 0; k < i; k++) {
                const int row_k = k * size;
                local_sum -= a[row_j + k] * a[row_k + i];
            }

            /* Write back result; division moved out of inner loop */
            a[row_j + i] = local_sum / pivot;
        }
    }
}
