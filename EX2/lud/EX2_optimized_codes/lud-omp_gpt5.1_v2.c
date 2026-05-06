#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        /* Upper triangular update: A[i, j] */
        for (j = i; j < size; j++) {
            float sum = a[i * size + j];
            /* Loop over k is sequential and typically small; keep as-is */
            for (k = 0; k < i; k++) {
                sum -= a[i * size + k] * a[k * size + j];
            }
            a[i * size + j] = sum;
        }

        /* Lower triangular update: A[j, i] */
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(a, size, i) private(j, k)
#endif
        for (j = i + 1; j < size; j++) {
            float sum = a[(size_t)j * size + i];
            for (k = 0; k < i; k++) {
                sum -= a[(size_t)j * size + k] * a[(size_t)k * size + i];
            }
            a[(size_t)j * size + i] = sum / a[(size_t)i * size + i];
        }
    }
}
