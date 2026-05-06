#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;
    printf("num of threads = %d\n", omp_num_threads);
    
    for (i = 0; i < size; i++) {
        float * __restrict__ row_i = &a[i * size];
        
        for (j = i; j < size; j++) {
            sum = row_i[j];
            for (k = 0; k < i; k++)
                sum -= row_i[k] * a[k * size + j];
            row_i[j] = sum;
        }
        
        float diag_inv = 1.0f / row_i[i];
        for (j = i + 1; j < size; j++) {
            float * __restrict__ row_j = &a[j * size];
            sum = row_j[i];
            for (k = 0; k < i; k++)
                sum -= row_j[k] * row_i[k];
            row_j[i] = sum * diag_inv;
        }
    }
}
