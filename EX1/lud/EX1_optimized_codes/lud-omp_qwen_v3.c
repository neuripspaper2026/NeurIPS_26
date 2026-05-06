#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;
    float *ptr_a_i, *ptr_a_k, *ptr_a_j;
    int size_i, size_k;
    
    printf("num of threads = %d\n", omp_num_threads);
    
    for (i = 0; i < size; i++) {
        ptr_a_i = a + i * size;
        size_i = i * size;
        
        for (j = i; j < size; j++) {
            sum = ptr_a_i[j];
            ptr_a_j = a + j * size;
            
            for (k = 0; k < i; k++) {
                size_k = k * size;
                sum -= (*(a + size_i + k)) * (*(a + size_k + j));
            }
            ptr_a_i[j] = sum;
        }
        
        for (j = i + 1; j < size; j++) {
            sum = (*(a + j * size + i));
            ptr_a_j = a + j * size;
            
            for (k = 0; k < i; k++) {
                size_k = k * size;
                sum -= (*(a + j * size + k)) * (*(a + size_k + i));
            }
            (*(a + j * size + i)) = sum / ptr_a_i[i];
        }
    }
}
