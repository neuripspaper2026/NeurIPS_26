#include <math.h>
#include <time.h>

static double step10_kernel_time = 0.0;

void reset_step10_kernel_time(void) { step10_kernel_time = 0.0; }
double get_step10_kernel_time(void) { return step10_kernel_time; }

void Step10_orig( int count1, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, float *xx1, float *yy1, float *zz1, float *mass1, float *dxi, float *dyi, float *dzi )
{
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const float ma0 = 0.269327f, ma1 = -0.0750978f, ma2 = 0.0114808f, ma3 = -0.00109313f, ma4 = 0.0000605491f, ma5 = -0.00000147177f;
    
    float dxc, dyc, dzc, m, r2, f, xi = 0.0f, yi = 0.0f, zi = 0.0f;
    float r2_plus_mp_rsm2, r2_plus_mp_rsm2_inv_sqrt;
    int j;

    for ( j = 0; j < count1; j++ ) 
    {
        dxc = xx1[j] - xxi;
        dyc = yy1[j] - yyi;
        dzc = zz1[j] - zzi;
  
        r2 = dxc * dxc + dyc * dyc + dzc * dzc;
       
        if (r2 <= 0.0f) continue;
        
        m = ( r2 < fsrrmax2 ) ? mass1[j] : 0.0f;

        r2_plus_mp_rsm2 = r2 + mp_rsm2;
        r2_plus_mp_rsm2_inv_sqrt = 1.0f / sqrtf(r2_plus_mp_rsm2);
        f = r2_plus_mp_rsm2_inv_sqrt * r2_plus_mp_rsm2_inv_sqrt * r2_plus_mp_rsm2_inv_sqrt;
        f = f - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5)))));

        f = m * f;

        xi += f * dxc;
        yi += f * dyc;
        zi += f * dzc;
    }

    *dxi = xi;
    *dyi = yi;
    *dzi = zi;

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    step10_kernel_time +=
        (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
        (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
