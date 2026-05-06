#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        const int row_i = i * size;

        /* Upper triangular update */
        for (j = i; j < size; j++) {
            float sum = a[row_i + j];
            float *row_i_base = a + row_i;
            float *col_j_base = a + j;
            for (k = 0; k < i; k++) {
                sum -= row_i_base[k] * col_j_base[k * size];
            }
            a[row_i + j] = sum;
        }

        /* Lower triangular update */
        const float diag = a[row_i + i];
        for (j = i + 1; j < size; j++) {
            const int row_j = j * size;
            float sum = a[row_j + i];
            float *row_j_base = a + row_j;
            float *col_i_base = a + i;
            for (k = 0; k < i; k++) {
                sum -= row_j_base[k] * col_i_base[k * size];
            }
            a[row_j + i] = sum / diag;
        }
    }
}
