#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        int const row_i = i * size;

        /* Upper triangular update */
        for (j = i; j < size; j++) {
            float sum = a[row_i + j];
            int col_j = j; /* base index for column j, row 0: 0*size + j */

            /* sum -= a[i*size + k] * a[k*size + j] */
            for (k = 0; k < i; k++) {
                int row_k = k * size;
                sum -= a[row_i + k] * a[row_k + col_j];
            }
            a[row_i + j] = sum;
        }

        /* Lower triangular update */
        for (j = i + 1; j < size; j++) {
            int row_j = j * size;
            float sum = a[row_j + i];

            /* sum -= a[j*size + k] * a[k*size + i] */
            for (k = 0; k < i; k++) {
                int row_k = k * size;
                sum -= a[row_j + k] * a[row_k + i];
            }
            a[row_j + i] = sum / a[row_i + i];
        }
    }
}
