#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);
    
    for (i = 0; i < size; i++) {
        // Upper triangular part - compute pivot row
        for (j = i; j < size; j++) {
            float sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        
        // Lower triangular part - parallelize across rows
        float pivot = a[i * size + i];
        #pragma omp parallel for schedule(static) if(size > 64)
        for (j = i + 1; j < size; j++) {
            float sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / pivot;
        }
    }
}
