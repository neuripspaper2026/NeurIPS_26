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
        int row_i = i * size;

        /* Upper part: U(i, j) for j = i..size-1 */
        for (j = i; j < size; j++) {
            int idx_ij = row_i + j;
            sum = a[idx_ij];
#pragma omp simd private(k) reduction(-:sum)
            for (k = 0; k < i; k++) {
                sum -= a[row_i + k] * a[k * size + j];
            }
            a[idx_ij] = sum;
        }

        /* Lower part: L(j, i) for j = i+1..size-1 */
        {
            float diag = a[row_i + i];
#pragma omp parallel for default(none) private(j, k, sum) shared(a, size, i, row_i, diag)
            for (j = i + 1; j < size; j++) {
                int row_j = j * size;
                int idx_ji = row_j + i;
                sum = a[idx_ji];
#pragma omp simd private(k) reduction(-:sum)
                for (k = 0; k < i; k++) {
                    sum -= a[row_j + k] * a[k * size + i];
                }
                a[idx_ji] = sum / diag;
            }
        }
    }
}
