#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        const int row_i = i * size;
        const float *const a_row_i = a + row_i;

        /* Upper triangular update (U part) */
        for (j = i; j < size; j++) {
            float sum = a_row_i[j];
#ifdef _OPENMP
#pragma omp simd reduction(-:sum)
#endif
            for (k = 0; k < i; k++) {
                sum -= a_row_i[k] * a[k * size + j];
            }
            a_row_i[j] = sum;
        }

        const float diag = a_row_i[i];

        /* Lower triangular update (L part) */
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(a, size, i, diag) private(j, k)
#endif
        for (j = i + 1; j < size; j++) {
            const int row_j = j * size;
            float *const a_row_j = a + row_j;
            float sum = a_row_j[i];
#ifdef _OPENMP
#pragma omp simd reduction(-:sum)
#endif
            for (k = 0; k < i; k++) {
                sum -= a_row_j[k] * a[k * size + i];
            }
            a_row_j[i] = sum / diag;
        }
    }
}
