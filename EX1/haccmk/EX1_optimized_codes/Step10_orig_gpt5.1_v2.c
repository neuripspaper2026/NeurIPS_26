#include <math.h>
#include <time.h>
#include <stddef.h>
#include <omp.h>

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

    const float xxi_local = xxi;
    const float yyi_local = yyi;
    const float zzi_local = zzi;
    const float fsrrmax2_local = fsrrmax2;
    const float mp_rsm2_local  = mp_rsm2;

    float xi = 0.0f;
    float yi = 0.0f;
    float zi = 0.0f;

    int j = 0;

    const int unroll_factor = 4;
    const int limit = count1 - (count1 % unroll_factor);

    for ( ; j < limit; j += unroll_factor )
    {
        float dxc0 = xx1[j]     - xxi_local;
        float dyc0 = yy1[j]     - yyi_local;
        float dzc0 = zz1[j]     - zzi_local;

        float dxc1 = xx1[j + 1] - xxi_local;
        float dyc1 = yy1[j + 1] - yyi_local;
        float dzc1 = zz1[j + 1] - zzi_local;

        float dxc2 = xx1[j + 2] - xxi_local;
        float dyc2 = yy1[j + 2] - yyi_local;
        float dzc2 = zz1[j + 2] - zzi_local;

        float dxc3 = xx1[j + 3] - xxi_local;
        float dyc3 = yy1[j + 3] - yyi_local;
        float dzc3 = zz1[j + 3] - zzi_local;

        float r20 = dxc0 * dxc0 + dyc0 * dyc0 + dzc0 * dzc0;
        float r21 = dxc1 * dxc1 + dyc1 * dyc1 + dzc1 * dzc1;
        float r22 = dxc2 * dxc2 + dyc2 * dyc2 + dzc2 * dzc2;
        float r23 = dxc3 * dxc3 + dyc3 * dyc3 + dzc3 * dzc3;

        float m0 = (r20 < fsrrmax2_local) ? mass1[j]     : 0.0f;
        float m1 = (r21 < fsrrmax2_local) ? mass1[j + 1] : 0.0f;
        float m2 = (r22 < fsrrmax2_local) ? mass1[j + 2] : 0.0f;
        float m3 = (r23 < fsrrmax2_local) ? mass1[j + 3] : 0.0f;

        float t0 = r20 + mp_rsm2_local;
        float t1 = r21 + mp_rsm2_local;
        float t2 = r22 + mp_rsm2_local;
        float t3 = r23 + mp_rsm2_local;

        float inv_r0 = 1.0f / sqrtf(t0);
        float inv_r1 = 1.0f / sqrtf(t1);
        float inv_r2 = 1.0f / sqrtf(t2);
        float inv_r3 = 1.0f / sqrtf(t3);

        float inv_r3_0 = inv_r0 * inv_r0 * inv_r0;
        float inv_r3_1 = inv_r1 * inv_r1 * inv_r1;
        float inv_r3_2 = inv_r2 * inv_r2 * inv_r2;
        float inv_r3_3 = inv_r3 * inv_r3 * inv_r3;

        float poly0 = ma0 + r20 * (ma1 + r20 * (ma2 + r20 * (ma3 + r20 * (ma4 + r20 * ma5))));
        float poly1 = ma0 + r21 * (ma1 + r21 * (ma2 + r21 * (ma3 + r21 * (ma4 + r21 * ma5))));
        float poly2 = ma0 + r22 * (ma1 + r22 * (ma2 + r22 * (ma3 + r22 * (ma4 + r22 * ma5))));
        float poly3 = ma0 + r23 * (ma1 + r23 * (ma2 + r23 * (ma3 + r23 * (ma4 + r23 * ma5))));

        float f0 = inv_r3_0 - poly0;
        float f1 = inv_r3_1 - poly1;
        float f2 = inv_r3_2 - poly2;
        float f3 = inv_r3_3 - poly3;

        f0 = (r20 > 0.0f) ? m0 * f0 : 0.0f;
        f1 = (r21 > 0.0f) ? m1 * f1 : 0.0f;
        f2 = (r22 > 0.0f) ? m2 * f2 : 0.0f;
        f3 = (r23 > 0.0f) ? m3 * f3 : 0.0f;

        xi += f0 * dxc0 + f1 * dxc1 + f2 * dxc2 + f3 * dxc3;
        yi += f0 * dyc0 + f1 * dyc1 + f2 * dyc2 + f3 * dyc3;
        zi += f0 * dzc0 + f1 * dzc1 + f2 * dzc2 + f3 * dzc3;
    }

    for ( ; j < count1; ++j )
    {
        float dxc = xx1[j] - xxi_local;
        float dyc = yy1[j] - yyi_local;
        float dzc = zz1[j] - zzi_local;

        float r2 = dxc * dxc + dyc * dyc + dzc * dzc;

        float m = (r2 < fsrrmax2_local) ? mass1[j] : 0.0f;

        float t = r2 + mp_rsm2_local;
        float inv_r = 1.0f / sqrtf(t);
        float inv_r3 = inv_r * inv_r * inv_r;

        float poly = ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))));

        float f = inv_r3 - poly;

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
