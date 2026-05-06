#include <math.h>
#include <time.h>

static double step10_kernel_time = 0.0;

void reset_step10_kernel_time(void) { step10_kernel_time = 0.0; }
double get_step10_kernel_time(void) { return step10_kernel_time; }

void Step10_orig( int count1, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, float *xx1, float *yy1, float *zz1, float *mass1, float *dxi, float *dyi, float *dzi )
{
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const float ma0 = 0.269327f;
    const float ma1 = -0.0750978f;
    const float ma2 = 0.0114808f;
    const float ma3 = -0.00109313f;
    const float ma4 = 0.0000605491f;
    const float ma5 = -0.00000147177f;

    float xi = 0.0f;
    float yi = 0.0f;
    float zi = 0.0f;

    const float xxi_local = xxi;
    const float yyi_local = yyi;
    const float zzi_local = zzi;
    const float fsrrmax2_local = fsrrmax2;
    const float mp_rsm2_local = mp_rsm2;

    for ( int j = 0; j < count1; j++ )
    {
        const float dxc = xx1[j] - xxi_local;
        const float dyc = yy1[j] - yyi_local;
        const float dzc = zz1[j] - zzi_local;

        const float r2 = dxc * dxc + dyc * dyc + dzc * dzc;

        const float m = ( r2 < fsrrmax2_local ) ? mass1[j] : 0.0f;

        const float r2_plus = r2 + mp_rsm2_local;
        const float inv_r = 1.0f / sqrtf( r2_plus );
        const float inv_r3 = inv_r * inv_r * inv_r;

        const float poly = ma0 + r2 * ( ma1 + r2 * ( ma2 + r2 * ( ma3 + r2 * ( ma4 + r2 * ma5 ) ) ) );

        float f = inv_r3 - poly;
        f = ( r2 > 0.0f ) ? m * f : 0.0f;

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
