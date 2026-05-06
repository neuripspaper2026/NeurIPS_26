#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;
    float *a_i, *a_k;  // Pointer to current row i and row k
    printf("num of threads = %d\n", omp_num_threads);
    for (i = 0; i < size; i++) {
        a_i = &a[i * size];  // Precompute pointer to row i
        for (j = i; j < size; j++) {
            sum = a_i[j];  // Use precomputed pointer
            for (k = 0; k < i; k++) {
                a_k = &a[k * size];  // Precompute pointer to row k
                sum -= a_i[k] * a_k[j];
            }
            a_i[j] = sum;
        }
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++) {
                a_k = &a[k * size];  // Reuse precomputed pointer
                sum -= a[j * size + k] * a_k[i];
            }
            a[j * size + i] = sum / a_i[i];
        }
    }
}
