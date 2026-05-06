#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;
    float *pivot_row, *current_row;
    int size_minus_1 = size - 1;
    
    printf("num of threads = %d\n", omp_num_threads);
    
    for (i = 0; i < size; i++) {
        pivot_row = &a[i * size];
        
        // Compute diagonal and upper triangular elements
        for (j = i; j < size; j++) {
            sum = pivot_row[j];
            for (k = 0; k < i; k++) {
                sum -= pivot_row[k] * a[k * size + j];
            }
            pivot_row[j] = sum;
        }
        
        // Compute lower triangular elements
        for (j = i + 1; j < size; j++) {
            current_row = &a[j * size];
            sum = current_row[i];
            for (k = 0; k < i; k++) {
                sum -= current_row[k] * a[k * size + i];
            }
            current_row[i] = sum / pivot_row[i];
        }
    }
}
