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
        /* U part: row i, columns i..size-1 */
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
#pragma omp simd reduction(-:sum)
            for (k = 0; k < i; k++) {
                sum -= a[i * size + k] * a[k * size + j];
            }
            a[i * size + j] = sum;
        }

        /* L part: column i, rows i+1..size-1 */
#pragma omp parallel for private(j, k, sum) default(none) shared(a, size, i)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
#pragma omp simd reduction(-:sum)
            for (k = 0; k < i; k++) {
                sum -= a[j * size + k] * a[k * size + i];
            }
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}
