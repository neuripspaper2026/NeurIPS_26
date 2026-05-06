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

    const float ma0 = 0.269327f,  ma1 = -0.0750978f, ma2 = 0.0114808f;
    const float ma3 = -0.00109313f, ma4 = 0.0000605491f, ma5 = -0.00000147177f;

    float xi = 0.0f;
    float yi = 0.0f;
    float zi = 0.0f;

    /* Use OpenMP to parallelize the main loop with reduction on xi, yi, zi.
       The math is associative/commutative enough for floating-point reductions. */
#ifdef _OPENMP
#pragma omp parallel
    {
        float xi_private = 0.0f;
        float yi_private = 0.0f;
        float zi_private = 0.0f;

#pragma omp for nowait
        for (int j = 0; j < count1; ++j)
        {
            const float dxc = xx1[j] - xxi;
            const float dyc = yy1[j] - yyi;
            const float dzc = zz1[j] - zzi;

            const float r2 = dxc * dxc + dyc * dyc + dzc * dzc;

            /* Fast path: if outside cutoff or zero radius, contribution is zero */
            if (r2 <= 0.0f || r2 >= fsrrmax2)
                continue;

            const float m = mass1[j];

            /* Use rsqrtf for 1/sqrt and then build (r2 + mp_rsm2)^(-1.5)
               as inv_r_sqrt^3 to avoid powf in the inner loop. */
            const float r2m = r2 + mp_rsm2;
            const float inv_r = 1.0f / sqrtf(r2m);
            const float inv_r3 = inv_r * inv_r * inv_r;

            /* Horner polynomial evaluation for the correction term */
            const float poly = ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))));
            float f = inv_r3 - poly;
            f *= m;

            xi_private += f * dxc;
            yi_private += f * dyc;
            zi_private += f * dzc;
        }

        /* Manual reduction to avoid false sharing on xi/yi/zi */
#pragma omp atomic
        xi += xi_private;
#pragma omp atomic
        yi += yi_private;
#pragma omp atomic
        zi += zi_private;
    }
#else
    for (int j = 0; j < count1; ++j)
    {
        const float dxc = xx1[j] - xxi;
        const float dyc = yy1[j] - yyi;
        const float dzc = zz1[j] - zzi;

        const float r2 = dxc * dxc + dyc * dyc + dzc * dzc;

        if (r2 <= 0.0f || r2 >= fsrrmax2)
            continue;

        const float m = mass1[j];

        const float r2m = r2 + mp_rsm2;
        const float inv_r = 1.0f / sqrtf(r2m);
        const float inv_r3 = inv_r * inv_r * inv_r;

        const float poly = ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))));
        float f = inv_r3 - poly;
        f *= m;

        xi += f * dxc;
        yi += f * dyc;
        zi += f * dzc;
    }
#endif

    *dxi = xi;
    *dyi = yi;
    *dzi = zi;

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    step10_kernel_time +=
        (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
        (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
