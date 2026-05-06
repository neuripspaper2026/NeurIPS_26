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
        /* U factor (row update) */
#ifdef _OPENMP
#pragma omp parallel for default(none) private(j, k, sum) shared(a, size, i) if(size > 64)
#endif
        for (j = i; j < size; j++) {
            float local_sum = a[i * size + j];
            int base_i = i * size;
            for (k = 0; k < i; k++) {
                local_sum -= a[base_i + k] * a[k * size + j];
            }
            a[base_i + j] = local_sum;
        }

        /* L factor (column update) */
#ifdef _OPENMP
#pragma omp parallel for default(none) private(j, k, sum) shared(a, size, i) if(size > 64)
#endif
        for (j = i + 1; j < size; j++) {
            float local_sum = a[j * size + i];
            int base_j = j * size;
            for (k = 0; k < i; k++) {
                local_sum -= a[base_j + k] * a[k * size + i];
            }
            a[base_j + i] = local_sum / a[i * size + i];
        }
    }
}
