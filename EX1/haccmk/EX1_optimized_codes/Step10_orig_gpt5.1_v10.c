#include <math.h>
#include <time.h>

static double step10_kernel_time = 0.0;

void reset_step10_kernel_time(void) { step10_kernel_time = 0.0; }
double get_step10_kernel_time(void) { return step10_kernel_time; }

void Step10_orig( int count1, float xxi, float yyi, float zzi, float fsrrmax2, float mp_rsm2, float *xx1, float *yy1, float *zz1, float *mass1, float *dxi, float *dyi, float *dzi )
{
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const float ma0 = 0.269327f, ma1 = -0.0750978f, ma2 = 0.0114808f,
                ma3 = -0.00109313f, ma4 = 0.0000605491f, ma5 = -0.00000147177f;

    float xi = 0.0f;
    float yi = 0.0f;
    float zi = 0.0f;

    const float local_xxi      = xxi;
    const float local_yyi      = yyi;
    const float local_zzi      = zzi;
    const float local_fsrrmax2 = fsrrmax2;
    const float local_mp_rsm2  = mp_rsm2;

    for ( int j = 0; j < count1; ++j )
    {
        const float dxc = xx1[j] - local_xxi;
        const float dyc = yy1[j] - local_yyi;
        const float dzc = zz1[j] - local_zzi;

        const float r2 = dxc * dxc + dyc * dyc + dzc * dzc;

        const float m = (r2 < local_fsrrmax2) ? mass1[j] : 0.0f;

        /* Compute (r2 + mp_rsm2)^(-1.5) as expf(-1.5 * logf(r2 + mp_rsm2)) */
        const float rp = r2 + local_mp_rsm2;
        const float inv_rp_1_5 = expf(-1.5f * logf(rp));

        const float r2_sq  = r2 * r2;
        const float r2_cub = r2_sq * r2;
        const float r2_4   = r2_sq * r2_sq;
        const float r2_5   = r2_4 * r2;

        const float poly = ma0
                         + ma1 * r2
                         + ma2 * r2_sq
                         + ma3 * r2_cub
                         + ma4 * r2_4
                         + ma5 * r2_5;

        float f = inv_rp_1_5 - poly;

        f = (r2 > 0.0f) ? m * f : 0.0f;

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
