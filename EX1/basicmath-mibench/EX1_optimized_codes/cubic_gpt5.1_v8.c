#include <math.h>
#include <time.h>
#include "../snipmath.h"

static double solve_cubic_kernel_time_acc = 0.0;

void reset_solve_cubic_kernel_time(void) { solve_cubic_kernel_time_acc = 0.0; }
double get_solve_cubic_kernel_time(void) { return solve_cubic_kernel_time_acc; }

void SolveCubic(double  a,
                double  b,
                double  c,
                double  d,
                int    *solutions,
                double *x)
{
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const long double inv_a = 1.0L / a;
    const long double a1 = b * inv_a;
    const long double a2 = c * inv_a;
    const long double a3 = d * inv_a;

    const long double a1_sq = a1 * a1;
    const long double Q = (a1_sq - 3.0L * a2) / 9.0L;
    const long double a1_cu = a1_sq * a1;
    const long double R = (2.0L * a1_cu - 9.0L * a1 * a2 + 27.0L * a3) / 54.0L;
    const long double Q3 = Q * Q * Q;
    const long double R2 = R * R;
    const long double R2_Q3 = R2 - Q3;

    if (R2_Q3 <= 0.0L)
    {
        *solutions = 3;
        const long double sqrtQ = sqrtl(Q);
        const long double Q3_sqrt = sqrtl(Q3);
        const long double ratio = R / Q3_sqrt;
        const long double theta = acosl(ratio);

        const long double two_sqrtQ = 2.0L * sqrtQ;
        const long double a1_div_3 = a1 / 3.0L;

        const long double theta_div_3 = theta / 3.0L;
        const long double theta_p2pi_div_3 = (theta + 2.0L * PI) / 3.0L;
        const long double theta_p4pi_div_3 = (theta + 4.0L * PI) / 3.0L;

        const long double c0 = cosl(theta_div_3);
        const long double c1 = cosl(theta_p2pi_div_3);
        const long double c2 = cosl(theta_p4pi_div_3);

        const long double base = -a1_div_3;

        x[0] = (double)(-two_sqrtQ * c0 + base);
        x[1] = (double)(-two_sqrtQ * c1 + base);
        x[2] = (double)(-two_sqrtQ * c2 + base);
    }
    else
    {
        *solutions = 1;
        const long double sqrtR2_Q3 = sqrtl(R2_Q3);
        const long double absR = fabsl(R);
        long double x0 = powl(sqrtR2_Q3 + absR, 1.0L / 3.0L);

        x0 += Q / x0;
        if (R >= 0.0L)
            x0 = -x0;

        x0 -= a1 / 3.0L;
        x[0] = (double)x0;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    solve_cubic_kernel_time_acc +=
        (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
        (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}

#ifdef TEST

int main(void)
{
      double  a1 = 1.0, b1 = -10.5, c1 = 32.0, d1 = -30.0;
      double  a2 = 1.0, b2 = -4.5, c2 = 17.0, d2 = -30.0;
      double  x[3];
      int     solutions;

      SolveCubic(a1, b1, c1, d1, &solutions, x);

      /* should get 3 solutions: 2, 6 & 2.5   */

      SolveCubic(a2, b2, c2, d2, &solutions, x);

      /* should get 1 solution: 2.5           */

      return 0;
}

#endif /* TEST */
