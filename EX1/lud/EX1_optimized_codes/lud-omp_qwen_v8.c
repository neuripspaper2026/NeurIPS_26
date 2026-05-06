#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;
    float *a_i, *a_k;  // Cache pointers for better locality
    printf("num of threads = %d\n", omp_num_threads);
    
    for (i = 0; i < size; i++) {
        a_i = &a[i * size];  // Cache row pointer
        
        // First loop: compute U matrix
        for (j = i; j < size; j++) {
            sum = a_i[j];  // Use cached pointer
            for (k = 0; k < i; k++) {
                a_k = &a[k * size];  // Cache k-th row pointer
                sum -= a_i[k] * a_k[j];
            }
            a_i[j] = sum;
        }
        
        // Second loop: compute L matrix
        float diag_inv = 1.0f / a_i[i];  // Cache diagonal inverse
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++) {
                a_k = &a[k * size];  // Cache k-th row pointer
                sum -= a[j * size + k] * a_k[i];
            }
            a[j * size + i] = sum * diag_inv;  // Use multiplication instead of division
        }
    }
}
