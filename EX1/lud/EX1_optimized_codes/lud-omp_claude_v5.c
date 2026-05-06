#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;
    printf("num of threads = %d\n", omp_num_threads);
    
    for (i = 0; i < size; i++) {
        float * restrict a_i = &a[i * size];
        
        for (j = i; j < size; j++) {
            sum = a_i[j];
            for (k = 0; k < i; k++)
                sum -= a_i[k] * a[k * size + j];
            a_i[j] = sum;
        }
        
        float a_ii = a_i[i];
        for (j = i + 1; j < size; j++) {
            float * restrict a_j = &a[j * size];
            sum = a_j[i];
            for (k = 0; k < i; k++)
                sum -= a_j[k] * a[k * size + i];
            a_j[i] = sum / a_ii;
        }
    }
}
