#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        /* Upper triangular update (row i, columns j>=i) */
        for (j = i; j < size; j++) {
            float sum = a[i * size + j];
            /* Use pointer arithmetic to reduce repeated index calculations */
            float *row_i      = a + i * size;
            float *col_j_base = a + j; /* will access as col_j_base[k*size] */

#ifdef _OPENMP
#pragma omp simd reduction(-:sum)
#endif
            for (k = 0; k < i; k++) {
                sum -= row_i[k] * col_j_base[k * size];
            }
            row_i[j] = sum;
        }

        /* Lower triangular update (column i, rows j>i) */
        {
            float diag = a[i * size + i];
            float *col_i = a + i; /* will access as col_i[row*size] */

            for (j = i + 1; j < size; j++) {
                float sum = a[j * size + i];
                float *row_j = a + j * size;

#ifdef _OPENMP
#pragma omp simd reduction(-:sum)
#endif
                for (k = 0; k < i; k++) {
                    sum -= row_j[k] * col_i[k * size];
                }
                row_j[i] = sum / diag;
            }
        }
    }
}
