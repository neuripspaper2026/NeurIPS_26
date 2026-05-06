#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        const int row_i = i * size;
        float a_ii;

        /* Upper triangular / diagonal update: U(i, j), j >= i */
        for (j = i; j < size; j++) {
            float sum = a[row_i + j];
            const int col_j = j; /* used as offset in k-loop */

            for (k = 0; k < i; k++) {
                const int row_k = k * size;
                sum -= a[row_i + k] * a[row_k + col_j];
            }
            a[row_i + j] = sum;
        }

        a_ii = a[row_i + i];

        /* Lower triangular update: L(j, i), j > i */
        for (j = i + 1; j < size; j++) {
            const int row_j = j * size;
            float sum = a[row_j + i];

            for (k = 0; k < i; k++) {
                const int row_k = k * size;
                sum -= a[row_j + k] * a[row_k + i];
            }
            a[row_j + i] = sum / a_ii;
        }
    }
}
