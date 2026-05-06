#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double step10_kernel_time = 0.0;

void reset_step10_kernel_time(void) { step10_kernel_time = 0.0; }
double get_step10_kernel_time(void) { return step10_kernel_time; }

void Step10_orig( int count1, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, float *xx1, float *yy1, float *zz1, float *mass1, float *dxi, float *dyi, float *dzi )
{
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;
    
    float dxc, dyc, dzc, m, r2, f, xi, yi, zi;
    int j;

    xi = 0.; yi = 0.; zi = 0.;

    #pragma omp simd reduction(+:xi,yi,zi)
    for ( j = 0; j < count1; j++ ) 
    {
        dxc = xx1[j] - xxi;
        dyc = yy1[j] - yyi;
        dzc = zz1[j] - zzi;
  
        r2 = dxc * dxc + dyc * dyc + dzc * dzc;
       
        m = ( r2 < fsrrmax2 ) ? mass1[j] : 0.0f;

        float r2_plus_mp_rsm2 = r2 + mp_rsm2;
        float r2_plus_mp_rsm2_pow = powf(r2_plus_mp_rsm2, -1.5f);
        float poly = ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))));
        
        f =  r2_plus_mp_rsm2_pow - poly;
        
        f = ( r2 > 0.0f ) ? m * f : 0.0f;

        xi = xi + f * dxc;
        yi = yi + f * dyc;
        zi = zi + f * dzc;
    }

    *dxi = xi;
    *dyi = yi;
    *dzi = zi;

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    step10_kernel_time +=
        (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
        (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
