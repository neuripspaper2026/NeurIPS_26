#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        const int row_i = i * size;

        /* Upper triangular part (including diagonal) */
        for (j = i; j < size; j++) {
            float sum = a[row_i + j];
            const int col_j = j; /* reuse j as column offset base */

            for (k = 0; k < i; k++) {
                const int row_i_k = row_i + k;
                const int row_k_j = k * size + col_j;
                sum -= a[row_i_k] * a[row_k_j];
            }
            a[row_i + j] = sum;
        }

        /* Lower triangular part (below diagonal) */
        const float diag = a[row_i + i];
        for (j = i + 1; j < size; j++) {
            const int row_j = j * size;
            float sum = a[row_j + i];

            for (k = 0; k < i; k++) {
                const int row_j_k = row_j + k;
                const int row_k_i = k * size + i;
                sum -= a[row_j_k] * a[row_k_i];
            }
            a[row_j + i] = sum / diag;
        }
    }
}
