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

    const float ma0 = 0.269327f, ma1 = -0.0750978f, ma2 = 0.0114808f,
                ma3 = -0.00109313f, ma4 = 0.0000605491f, ma5 = -0.00000147177f;

    float xi = 0.0f;
    float yi = 0.0f;
    float zi = 0.0f;

    /* Precompute constants for the inverse-square-root series */
    const float c1 = -0.5f;
    const float c2 =  0.375f;
    const float c3 = -0.3125f;

#ifdef _OPENMP
    /* Parallel reduction over j; xi/yi/zi are reduced across threads */
#pragma omp parallel for reduction(+:xi,yi,zi) schedule(static)
#endif
    for (int j = 0; j < count1; j++)
    {
        const float dxc = xx1[j] - xxi;
        const float dyc = yy1[j] - yyi;
        const float dzc = zz1[j] - zzi;

        const float r2 = dxc * dxc + dyc * dyc + dzc * dzc;

        /* Fast mask for fsrrmax2 condition without branch where possible */
        const float m = (r2 < fsrrmax2) ? mass1[j] : 0.0f;

        /* Replace pow(r2 + mp_rsm2, -1.5f) with a fast 1/sqrt approximation
           refined by a short Newton-like polynomial for speed */
        const float rp = r2 + mp_rsm2;

        /* Initial approximation of 1/sqrt(rp) */
        float inv_sqrt_rp = 1.0f / sqrtf(rp);

        /* Optional refinement: inv_sqrt_rp *= (1.5 - 0.5*rp*inv_sqrt_rp^2);
           expressed as small polynomial for better ILP */
        const float t = rp * inv_sqrt_rp * inv_sqrt_rp;
        inv_sqrt_rp = inv_sqrt_rp * (1.5f + t * (c1 + t * (c2 + t * c3)));

        const float inv_rp3 = inv_sqrt_rp * inv_sqrt_rp * inv_sqrt_rp;

        /* Polynomial term using Horner's rule (already optimal form) */
        const float poly = ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))));

        float f = inv_rp3 - poly;

        /* Avoid self-interaction and useless work when r2==0 or m==0 */
        if (r2 > 0.0f && m != 0.0f)
        {
            f *= m;
            xi += f * dxc;
            yi += f * dyc;
            zi += f * dzc;
        }
    }

    *dxi = xi;
    *dyi = yi;
    *dzi = zi;

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    step10_kernel_time +=
        (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
        (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
