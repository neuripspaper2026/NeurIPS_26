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
        /* U-factor: row i, columns i..size-1 (upper triangular, including diagonal) */
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            /* Inner product: row i (0..i-1) · column j (0..i-1) */
            for (k = 0; k < i; k++) {
                sum -= a[i * size + k] * a[k * size + j];
            }
            a[i * size + j] = sum;
        }

        /* L-factor: column i, rows i+1..size-1 (strictly lower triangular) */
#ifdef _OPENMP
        #pragma omp parallel for default(none) private(j, k, sum) shared(a, size, i)
#endif
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            /* Inner product: row j (0..i-1) · column i (0..i-1) */
            for (k = 0; k < i; k++) {
                sum -= a[j * size + k] * a[k * size + i];
            }
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}
