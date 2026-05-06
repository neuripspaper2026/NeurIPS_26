#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    printf("num of threads = %d\n", omp_num_threads);

    for (int i = 0; i < size; ++i) {
        const int row_i_offset = i * size;

        /* Upper (including diagonal) part: U(i, j) */
        for (int j = i; j < size; ++j) {
            float sum = a[row_i_offset + j];
            const int col_j_offset = j; /* used for k*size + j */

            /* sum -= L(i, k) * U(k, j) for k = 0..i-1 */
            for (int k = 0; k < i; ++k) {
                const int row_k_offset = k * size;
                sum -= a[row_i_offset + k] * a[row_k_offset + col_j_offset];
            }

            a[row_i_offset + j] = sum;
        }

        /* Lower part: L(j, i) */
        const float pivot = a[row_i_offset + i];
        const int col_i_offset = i; /* used for k*size + i */

        for (int j = i + 1; j < size; ++j) {
            const int row_j_offset = j * size;
            float sum = a[row_j_offset + i];

            /* sum -= L(j, k) * U(k, i) for k = 0..i-1 */
            for (int k = 0; k < i; ++k) {
                const int row_k_offset = k * size;
                sum -= a[row_j_offset + k] * a[row_k_offset + col_i_offset];
            }

            a[row_j_offset + i] = sum / pivot;
        }
    }
}
