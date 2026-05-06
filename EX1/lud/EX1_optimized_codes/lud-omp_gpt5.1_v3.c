#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; ++i) {
        float *const row_i = a + (size_t)i * (size_t)size;
        const int i_plus_1 = i + 1;

        /* Upper triangular (U) */
        for (j = i; j < size; ++j) {
            float sum = row_i[j];
            const int limit = i;
            for (k = 0; k < limit; ++k) {
                const float aik = row_i[k];
                const float *const row_k = a + (size_t)k * (size_t)size;
                sum -= aik * row_k[j];
            }
            row_i[j] = sum;
        }

        /* Lower triangular (L) */
        for (j = i_plus_1; j < size; ++j) {
            float *const row_j = a + (size_t)j * (size_t)size;
            float sum = row_j[i];
            const int limit = i;
            for (k = 0; k < limit; ++k) {
                const float ajk = row_j[k];
                const float *const row_k = a + (size_t)k * (size_t)size;
                sum -= ajk * row_k[i];
            }
            row_j[i] = sum / row_i[i];
        }
    }
}
