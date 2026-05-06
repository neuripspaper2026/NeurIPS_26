#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; i++) {
        /* U part: a[i, j] for j >= i */
        for (j = i; j < size; j++) {
            float sum = a[i * size + j];
            int base_i = i * size;
            for (k = 0; k < i; k++) {
                sum -= a[base_i + k] * a[k * size + j];
            }
            a[base_i + j] = sum;
        }

        /* L part: a[j, i] for j > i */
#ifdef _OPENMP
        #pragma omp parallel for default(none) shared(a, size, i) private(j, k)
#endif
        for (j = i + 1; j < size; j++) {
            float sum = a[j * size + i];
            int base_j = j * size;
            for (k = 0; k < i; k++) {
                sum -= a[base_j + k] * a[k * size + i];
            }
            a[base_j + i] = sum / a[i * size + i];
        }
    }
}
