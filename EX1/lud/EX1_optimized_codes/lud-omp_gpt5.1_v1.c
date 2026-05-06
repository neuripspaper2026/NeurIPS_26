#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        const int row_i_offset = i * size;

        /* Upper triangular / diagonal update: a[i, j] */
        for (j = i; j < size; j++) {
            float sum = a[row_i_offset + j];
            int k_end = i;

            for (k = 0; k < k_end; k++) {
                const int row_k_offset = k * size;
                sum -= a[row_i_offset + k] * a[row_k_offset + j];
            }

            a[row_i_offset + j] = sum;
        }

        /* Lower triangular update: a[j, i] */
        const float diag_ii = a[row_i_offset + i];
        for (j = i + 1; j < size; j++) {
            const int row_j_offset = j * size;
            float sum = a[row_j_offset + i];
            int k_end = i;

            for (k = 0; k < k_end; k++) {
                sum -= a[row_j_offset + k] * a[(k * size) + i];
            }

            a[row_j_offset + i] = sum / diag_ii;
        }
    }
}
