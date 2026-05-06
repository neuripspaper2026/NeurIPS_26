#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        const int row_i_off = i * size;

        /* Upper-triangular/update U: a[i, j], j = i..size-1 */
        for (j = i; j < size; j++) {
            float sum = a[row_i_off + j];
            const int col_j_off = j; /* i * size already folded into row_i_off */
            for (k = 0; k < i; k++) {
                const int row_k_off = k * size;
                sum -= a[row_i_off + k] * a[row_k_off + col_j_off];
            }
            a[row_i_off + j] = sum;
        }

        /* Lower-triangular/update L: a[j, i], j = i+1..size-1 */
        const float diag_ii = a[row_i_off + i];
        for (j = i + 1; j < size; j++) {
            const int row_j_off = j * size;
            float sum = a[row_j_off + i];
            for (k = 0; k < i; k++) {
                const int row_k_off = k * size;
                sum -= a[row_j_off + k] * a[row_k_off + i];
            }
            a[row_j_off + i] = sum / diag_ii;
        }
    }
}
