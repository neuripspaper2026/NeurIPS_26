#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{
    const int nx_ny = nx * ny;
    const int c1_mult = c1 * 6; // Precompute c1 * 6 since we're adding 6 terms
    const float c0_neg = -c0;   // Precompute -c0 for subtraction

    // Use restrict keyword to inform compiler that pointers don't alias
    const float *restrict A0_ptr = A0;
    float *restrict Anext_ptr = Anext;

    for(int i = 1; i < nx - 1; i++)
    {
        const int i_ny = i * ny;
        for(int j = 1; j < ny - 1; j++)
        {
            const int base_index = i_ny + j;
            float *Anext_row = Anext_ptr + base_index * nz;
            const float *A0_row = A0_ptr + base_index * nz;

            for(int k = 1; k < nz - 1; k++)
            {
                // Compute indices more efficiently by using base_index
                const int curr = base_index * nz + k;
                const int next_k = curr + 1;
                const int prev_k = curr - 1;
                const int next_j = curr + nz;
                const int prev_j = curr - nz;
                const int next_i = curr + nx_ny;
                const int prev_i = curr - nx_ny;

                Anext_row[k] = 
                    (A0_row[k + 1] +
                     A0_row[k - 1] +
                     A0_ptr[prev_j + k] +
                     A0_ptr[next_j + k] +
                     A0_ptr[prev_i + k] +
                     A0_ptr[next_i + k]) * c1
                    + A0_ptr[curr] * c0_neg;
            }
        }
    }
}


