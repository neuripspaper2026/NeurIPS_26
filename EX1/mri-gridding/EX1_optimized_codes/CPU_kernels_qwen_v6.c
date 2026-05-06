<<<CODE>>>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359f

static inline float kernel_value_CPU(float v) {
    const float z = v * v;
    
    // Use Horner's method for polynomial evaluation to reduce operations
    const float num_coeffs[] = {
        0.210580722890567e-22f, 0.380715242345326e-19f, 0.479440257548300e-16f,
        0.435125971262668e-13f, 0.300931127112960e-10f, 0.160224679395361e-7f,
        0.654858370096785e-5f,  0.202591084143397e-2f,  0.463076284721000e0f,
        0.754337328948189e2f,   0.830792541809429e4f,   0.571661130563785e6f,
        0.216415572361227e8f,   0.356644482244025e9f,   0.144048298227235e10f
    };
    
    float num = num_coeffs[14];
    for (int i = 13; i >= 0; i--) {
        num = num * z + num_coeffs[i];
    }

    const float den = z * (z * (z - 0.307646912682801e4f) + 0.347626332405882e7f) - 0.144048298227235e10f;

    return -num / den;
}

void calculateLUT(float beta, float width, float** LUT, unsigned int* sizeLUT) {
    if (width <= 0) return;
    
    // compute size of LUT based on kernel width
    const unsigned int size = (unsigned int)(10000 * width);
    
    // allocate memory
    (*LUT) = (float*)malloc(size * sizeof(float));
    if (!(*LUT)) return;
    
    const float cutoff2 = (width * width) / 4.0f;
    const float inv_size = cutoff2 / (float)size;
    
    // Precompute beta and cutoff2 relationship
    const float beta_factor = beta * sqrtf(1.0f);
    
    for (unsigned int k = 0; k < size; ++k) {
        // compute value to evaluate kernel at
        const float v = ((float)k) * inv_size;
        
        // compute kernel value and store
        const float sqrt_arg = 1.0f - (v / cutoff2);
        (*LUT)[k] = kernel_value_CPU(beta * sqrtf(fmaxf(sqrt_arg, 0.0f)));
    }
    (*sizeLUT) = size;
}

static inline float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1overCutoff2) {
    const float scaled_v = v * (float)sizeLUT * _1overCutoff2;
    const unsigned int k0 = (unsigned int)scaled_v;
    const float v0 = ((float)k0) / _1overCutoff2;
    return LUT[k0] + ((scaled_v - (float)k0) * (LUT[k0 + 1] - LUT[k0]) * _1overCutoff2);
}

int gridding_Gold(unsigned int n, parameters params, ReconstructionSample* sample, float* LUT, unsigned int sizeLUT, cmplx* gridData, float* sampleDensity) {
    const unsigned int size_x = params.gridSize[0];
    const unsigned int size_y = params.gridSize[1];
    const unsigned int size_z = params.gridSize[2];

    const float cutoff = ((float)(params.kernelWidth)) / 2.0f;  // cutoff radius
    const float cutoff2 = cutoff * cutoff;                      // square of cutoff radius
    const float _1overCutoff2 = 1.0f / cutoff2;                 // 1 over square of cutoff radius

    const float beta = PI * sqrtf(4.0f * params.kernelWidth * params.kernelWidth / (params.oversample * params.oversample) * (params.oversample - 0.5f) * (params.oversample - 0.5f) - 0.8f);
    
    const int max_radius = (int)(cutoff + 1);
    
    // Precompute frequently used values
    const float size_xy = (float)(size_x * size_y);
    
    for (unsigned int i = 0; i < n; i++) {
        const ReconstructionSample pt = sample[i];
        
        // Skip zero samples early
        if ((pt.real == 0.0f && pt.imag == 0.0f) || pt.sdc == 0.0f)
            continue;

        const float kx = pt.kX;
        const float ky = pt.kY;
        const float kz = pt.kZ;

        const unsigned int NxL = max((kx - cutoff), 0.0f);
        const unsigned int NxH = min((kx + cutoff), (float)(size_x - 1));
        
        const unsigned int NyL = max((ky - cutoff), 0.0f);
        const unsigned int NyH = min((ky + cutoff), (float)(size_y - 1));
        
        const unsigned int NzL = max((kz - cutoff), 0.0f);
        const unsigned int NzH = min((kz + cutoff), (float)(size_z - 1));

        // Precompute dx2 values
        float Dx2[200];  // Increased size to handle larger kernels safely
        for (int nx = NxL; nx <= (int)NxH; ++nx) {
            const float diff = kx - nx;
            Dx2[nx - NxL] = diff * diff;
        }
        
        // Precompute dy2 values
        float Dy2[200];
        for (int ny = NyL; ny <= (int)NyH; ++ny) {
            const float diff = ky - ny;
            Dy2[ny - NyL] = diff * diff;
        }
        
        // Precompute dz2 values
        float Dz2[200];
        for (int nz = NzL; nz <= (int)NzH; ++nz) {
            const float diff = kz - nz;
            Dz2[nz - NzL] = diff * diff;
        }

        const unsigned int dz2_count = NzH - NzL + 1;
        const unsigned int dy2_count = NyH - NyL + 1;
        const unsigned int dx2_count = NxH - NxL + 1;
        
        unsigned int idxZ = (NzL - 1) * size_x * size_y;
        
        for (unsigned int dz_idx = 0; dz_idx < dz2_count; ++dz_idx) {
            const float dz2_val = Dz2[dz_idx];
            idxZ += size_x * size_y;
            
            // Early exit if too far in Z direction
            if (dz2_val >= cutoff2) continue;
            
            unsigned int idxY = (NyL - 1) * size_x;
            
            for (unsigned int dy_idx = 0; dy_idx < dy2_count; ++dy_idx) {
                const float dy2_val = Dy2[dy_idx];
                idxY += size_x;
                
                const float dy2dz2 = dz2_val + dy2_val;
                
                // Early exit if too far in Y direction
                if (dy2dz2 >= cutoff2) continue;
                
                const unsigned int idx0 = idxY + idxZ;
                
                for (unsigned int dx_idx = 0; dx_idx < dx2_count; ++dx_idx) {
                    const float dx2_val = Dx2[dx_idx];
                    const float v = dy2dz2 + dx2_val;
                    
                    if (v < cutoff2) {
                        const unsigned int nx = NxL + dx_idx;
                        const unsigned int idx = nx + idx0;
                        
                        float w;
                        if (params.useLUT) {
                            w = kernel_value_LUT(v
