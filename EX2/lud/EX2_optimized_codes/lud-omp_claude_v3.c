#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);
    
    for (i = 0; i < size; i++) {
        /* Upper triangular part - compute row i from column i onwards */
        for (j = i; j < size; j++) {
            float sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        
        /* Lower triangular part - compute column i from row i+1 onwards */
        float diag_inv = 1.0f / a[i * size + i];
        #pragma omp parallel for schedule(static) if(size - i - 1 > 32)
        for (j = i + 1; j < size; j++) {
            float sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum * diag_inv;
        }
    }
}
