#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        float * const a_row_i = a + (size * i);
        const int base_diag = i * size;

        /* Upper triangular part (including diagonal): parallel over columns */
        #pragma omp parallel for private(j, k, sum) schedule(static) if(size > 64)
        for (j = i; j < size; j++) {
            sum = a_row_i[j];
            const int col_index = j;
            for (k = 0; k < i; k++) {
                sum -= a_row_i[k] * a[k * size + col_index];
            }
            a_row_i[j] = sum;
        }

        /* Lower triangular part: parallel over rows */
        #pragma omp parallel for private(j, k, sum) schedule(static) if(size > 64)
        for (j = i + 1; j < size; j++) {
            float * const a_row_j = a + (size * j);
            sum = a_row_j[i];
            for (k = 0; k < i; k++) {
                sum -= a_row_j[k] * a[k * size + i];
            }
            a_row_j[i] = sum / a_row_i[i];
        }
    }
}
