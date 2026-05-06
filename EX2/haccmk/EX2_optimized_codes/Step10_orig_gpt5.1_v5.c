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

    const float ma0 = 0.269327f;
    const float ma1 = -0.0750978f;
    const float ma2 = 0.0114808f;
    const float ma3 = -0.00109313f;
    const float ma4 = 0.0000605491f;
    const float ma5 = -0.00000147177f;

    float xi = 0.0f;
    float yi = 0.0f;
    float zi = 0.0f;

    const float c1p5 = -1.5f;

#ifdef _OPENMP
    /* Parallelize outer loop with private accumulators and reduction */
    #pragma omp parallel
    {
        float local_xi = 0.0f;
        float local_yi = 0.0f;
        float local_zi = 0.0f;

        #pragma omp for nowait
        for (int j = 0; j < count1; j++)
        {
            const float dxc = xx1[j] - xxi;
            const float dyc = yy1[j] - yyi;
            const float dzc = zz1[j] - zzi;

            const float r2 = dxc * dxc + dyc * dyc + dzc * dzc;

            /* Fast conditional mass selection */
            const float m = (r2 < fsrrmax2) ? mass1[j] : 0.0f;

            /* Compute inverse distance approximation via exp/log to avoid powf */
            const float rp = r2 + mp_rsm2;
            const float inv_sqrt = expf(c1p5 * logf(rp));

            /* Polynomial evaluation with Horner's scheme (already optimal) */
            const float poly = ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))));
            float f = inv_sqrt - poly;

            /* Guard against r2 == 0 */
            f = (r2 > 0.0f) ? m * f : 0.0f;

            local_xi += f * dxc;
            local_yi += f * dyc;
            local_zi += f * dzc;
        }

        #pragma omp atomic
        xi += local_xi;
        #pragma omp atomic
        yi += local_yi;
        #pragma omp atomic
        zi += local_zi;
    }
#else
    for (int j = 0; j < count1; j++)
    {
        const float dxc = xx1[j] - xxi;
        const float dyc = yy1[j] - yyi;
        const float dzc = zz1[j] - zzi;

        const float r2 = dxc * dxc + dyc * dyc + dzc * dzc;

        const float m = (r2 < fsrrmax2) ? mass1[j] : 0.0f;

        const float rp = r2 + mp_rsm2;
        const float inv_sqrt = expf(c1p5 * logf(rp));

        const float poly = ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))));
        float f = inv_sqrt - poly;

        f = (r2 > 0.0f) ? m * f : 0.0f;

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
